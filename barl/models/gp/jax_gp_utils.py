"""
In order to take advantage of jax's vectorization and
JIT compilation, I thought it would be good to make a
functional decoupled GP implementation.

This was initially designed for use by the JointSetBaxAcqFunction
and still play nicely with the gp_params used by the BARL library
"""
import jax.numpy as jnp
from jax.scipy.linalg import solve_triangular
from functools import partial
import copy
import collections


def kern_exp_quad_ard(xmat1, xmat2, ls, alpha):
    """
    Exponentiated quadratic kernel function with
    dimensionwise lengthscales if ls is an ndarray.
    """
    xmat1 = jnp.expand_dims(xmat1, axis=1)
    xmat2 = jnp.expand_dims(xmat2, axis=0)
    diff = xmat1 - xmat2
    diff /= ls
    norm = jnp.sum(diff**2, axis=-1) / 2.0
    kern = alpha**2 * jnp.exp(-norm)
    return kern


def get_gp_params_list(gp_params, n_dimy):
    """
    Return list of gp_params dicts by parsing gp_params.

    Copied and modified to be a pure function from MultiGpfsGp.
    """
    gp_params_list = [copy.deepcopy(gp_params) for _ in range(n_dimy)]

    hyps = ["ls", "alpha", "sigma"]
    for hyp in hyps:
        if not isinstance(gp_params.get(hyp, 1), (float, int)):
            # If hyp exists in dict, and is not (float, int), assume is list of hyp
            for idx, gpp in enumerate(gp_params_list):
                gpp[hyp] = gp_params[hyp][idx]

    return gp_params_list


def construct_jax_kernels(params):
    """
    make kernel functions for JAX from BARL gp params
    """
    assert (
        params.get("kernel_str", "rbf") == "rbf"
    ), "rbf is the only supported kernel right now"
    param_list = get_gp_params_list(params["gp_params"], params["n_dimy"])
    kernels = []
    # this part copied from MultiGpfsGP
    for params in param_list:
        if not isinstance(params["ls"], collections.abc.Sequence):
            ls = jnp.array([params.ls for _ in range(params.n_dimx)])
        else:
            ls = jnp.array(params["ls"])
        kernel = partial(kern_exp_quad_ard, ls=ls, alpha=params["alpha"])
        kernels.append(kernel)
    return kernels


def get_pred_covs(x_data, y_data, x_pred, lmats, smats, kernels):
    covs = []
    for y, kernel, smat, lmat in zip(y_data.T, kernels, smats, lmats):
        cov = get_pred_cov(x_data, y, x_pred, smat, lmat, kernel)
        covs.append(cov)
    return jnp.stack(covs)


def get_pred_cov(x_data, y_data, x_pred, smat, lmat, kernel):
    k21 = kernel(x_pred, x_data)
    k22 = kernel(x_pred, x_pred)
    vmat = solve_lower_triangular(lmat, k21.T)
    k2 = k22 - vmat.T @ vmat
    return k2


def get_cholesky_decomp(k11_nonoise, sigma):
    # this is gonna be naive at first
    k11 = k11_nonoise + sigma**2 * jnp.eye(k11_nonoise.shape[0])
    return jnp.linalg.cholesky(k11)


def get_lmat_smat(x, y, kernel, sigma):
    k11_nonoise = kernel(x, x)
    while True:
        lmat = get_cholesky_decomp(k11_nonoise, sigma)
        if jnp.isnan(lmat).any():
            sigma *= 2
        else:
            break
    smat = solve_upper_triangular(lmat.T, solve_lower_triangular(lmat, y))
    return lmat, smat


def get_lmats_smats(x, y, kernels, sigma):
    lmats = []
    smats = []
    for i, kernel in enumerate(kernels):
        lmat, smat = get_lmat_smat(x, y[:, i], kernel, sigma)
        lmats.append(lmat)
        smats.append(smat)
    return jnp.stack(lmats), jnp.stack(smats)


def solve_lower_triangular(amat, b):
    """Solves amat*x=b when amat is lower triangular."""
    return solve_triangular_base(amat, b, lower=True)


def solve_upper_triangular(amat, b):
    """Solves amat*x=b when amat is upper triangular."""
    return solve_triangular_base(amat, b, lower=False)


def solve_triangular_base(amat, b, lower):
    """Solves amat*x=b when amat is a triangular matrix."""
    if amat.size == 0 and b.shape[0] == 0:
        return jnp.zeros((b.shape))
    else:
        return solve_triangular(amat, b, lower=lower)
