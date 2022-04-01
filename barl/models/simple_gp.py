"""
Code for Gaussian processes.
"""

from argparse import Namespace
import copy
import collections.abc
import numpy as np
import tensorflow as tf

from .gp.gp_utils import (
    kern_exp_quad_ard,
    tf_kern_exp_quad_ard,
    kern_exp_quad_ard_per,
    sample_mvn,
    gp_post,
    solve_lower_triangular,
    solve_upper_triangular,
    tf_solve_lower_triangular,
    tf_solve_upper_triangular,
    get_cholesky_decomp,
    tf_get_cholesky_decomp,
)
from ..util.base import Base
from ..util.misc_util import dict_to_namespace


class SimpleGp(Base):
    """
    Simple GP model without external backend.
    """

    def __init__(self, params=None, data=None, verbose=True, lmat=None, smat=None):
        """
        Parameters
        ----------
        params : Namespace_or_dict
            Namespace or dict of parameters for this model.
        data : Namespace_or_dict
            Namespace or dict of initial data, containing lists x and y.
        verbose : bool
            If True, print description string.
        """
        super().__init__(params, verbose)
        self.set_data(data, lmat=lmat, smat=smat)

    def set_params(self, params):
        """Set self.params, the parameters for this model."""
        super().set_params(params)
        params = dict_to_namespace(params)

        self.params.name = getattr(params, "name", "SimpleGp")
        self.params.n_dimx = getattr(params, "n_dimx", 2)
        self.params.ls = getattr(params, "ls", 3.7)
        self.params.alpha = getattr(params, "alpha", 1.85)
        self.params.sigma = getattr(params, "sigma", 1e-2)

        # Format lengthscale
        if not isinstance(self.params.ls, collections.abc.Sequence):
            self.params.ls = [self.params.ls for _ in range(self.params.n_dimx)]
        self.params.ls = np.array(self.params.ls).reshape(-1)

        # Set kernel
        self.set_kernel(params)

    def set_kernel(self, params):
        """Set self.params.kernel."""
        self.params.kernel_str = getattr(params, "kernel_str", "rbf")

        if self.params.kernel_str == "rbf":
            self.params.kernel = kern_exp_quad_ard

        elif self.params.kernel_str == "rbf_periodic":
            pdims = params.periodic_dims
            period = params.period

            def kern(xmat1, xmat2, ls, alpha):
                """Periodic rbf ard kernel with standardized format."""
                return kern_exp_quad_ard_per(
                    xmat1, xmat2, ls, alpha, pdims=pdims, period=period
                )

            self.params.kernel = kern

    def set_data(self, data, lmat=None, smat=None):
        """Set self.data."""
        if data is None:
            # Initialize self.data to be empty
            self.data = Namespace()
            self.data.x = []
            self.data.y = []
        else:
            data = dict_to_namespace(data)
            self.data = copy.deepcopy(data)

            x_train = self.data.x
            y_train = self.data.y
            kernel = self.params.kernel
            ls = self.params.ls
            alpha = self.params.alpha
            sigma = self.params.sigma

            assert (lmat is None) == (smat is None)
            if lmat is not None:
                self.lmat = lmat
                self.smat = smat
            else:
                k11_nonoise = kernel(x_train, x_train, ls, alpha)
                self.lmat = get_cholesky_decomp(k11_nonoise, sigma, "try_first")
                self.smat = solve_upper_triangular(
                    self.lmat.T, solve_lower_triangular(self.lmat, y_train)
                )

    def get_prior_mu_cov(self, x_list, full_cov=True):
        """
        Return GP prior parameters: mean (mu) and covariance (cov).

        Parameters
        ----------
        x_list : list
            List of numpy ndarrays, each representing a domain point.
        full_cov : bool
            If True, return covariance matrix. If False, return list of standard
            deviations.

        Returns
        -------
        mu : ndarray
            A numpy 1d ndarray with len=len(x_list) of floats, corresponding to
            posterior mean for each x in x_list.
        cov : ndarray
            If full_cov is False, return a numpy 1d ndarray with len=len(x_list) of
            floats, corresponding to posterior standard deviations for each x in x_list.
            If full_cov is True, return the covariance matrix as a numpy ndarray
            (len(x_list) x len(x_list)).
        """
        # NOTE: currently assumes constant zero prior mean function.
        # TODO: support other mean functions.
        mu = np.zeros(len(x_list))
        cov = self.params.kernel(x_list, x_list, self.params.ls, self.params.alpha)

        if full_cov is False:
            cov = np.sqrt(np.diag(cov))

        return mu, cov

    def get_post_mu_cov(self, x_list, full_cov=True):
        """
        Return GP posterior parameters: mean (mu) and covariance (cov). If there is no
        data, return the GP prior parameters.

        Parameters
        ----------
        x_list : list
            List of numpy ndarrays, each representing a domain point.
        full_cov : bool
            If True, return covariance matrix. If False, return list of standard
            deviations.

        Returns
        -------
        mu : ndarray
            A numpy 1d ndarray with len=len(x_list) of floats, corresponding to
            posterior mean for each x in x_list.
        cov : ndarray
            If full_cov is False, return a numpy 1d ndarray with len=len(x_list) of
            floats, corresponding to posterior standard deviations for each x in x_list.
            If full_cov is True, return the covariance matrix as a numpy ndarray
            (len(x_list) x len(x_list)).
        """
        k21 = self.params.kernel(x_list, self.data.x, self.params.ls, self.params.alpha)
        mu2 = k21.dot(self.smat)

        k22 = self.params.kernel(x_list, x_list, self.params.ls, self.params.alpha)
        vmat = solve_lower_triangular(self.lmat, k21.T)
        k2 = k22 - vmat.T.dot(vmat)
        if full_cov is False:
            k2 = np.sqrt(np.diag(k2))

        # Return mean and cov matrix (or std-dev array if full_cov=False)
        return mu2, k2

    def gp_post_wrapper(self, x_list, data, full_cov=True):
        """Wrapper for gp_post given a list of x and data Namespace."""
        if len(data.x) == 0:
            return self.get_prior_mu_cov(x_list, full_cov)
        # If data is not empty:
        mu, cov = gp_post(
            data.x,
            data.y,
            x_list,
            self.params.ls,
            self.params.alpha,
            self.params.sigma,
            self.params.kernel,
            full_cov=full_cov,
        )
        # Return mean and cov matrix (or std-dev array if full_cov=False)
        return mu, cov

    def get_post_mu_cov_single(self, x):
        """Get GP posterior for an input x. Return posterior mean and std for x."""
        mu_arr, std_arr = self.get_post_mu_cov([x], full_cov=False)
        return mu_arr[0], std_arr[0]

    def sample_prior_list(self, x_list, n_samp, full_cov=True):
        """Get samples from gp prior for each input in x_list."""
        mu, cov = self.get_prior_mu_cov(x_list, full_cov)
        return self.get_normal_samples(mu, cov, n_samp, full_cov)

    def sample_prior(self, x, n_samp):
        """Get samples from gp prior for input x."""
        sample_list = self.sample_prior_list([x], n_samp)
        return sample_list[0]

    def sample_post_list(self, x_list, n_samp, full_cov=True):
        """Get samples from gp posterior for each input in x_list."""
        if len(self.data.x) == 0:
            return self.sample_prior_list(x_list, n_samp, full_cov)

        # If data is not empty:
        mu, cov = self.get_post_mu_cov(x_list, full_cov)
        return self.get_normal_samples(mu, cov, n_samp, full_cov)

    def sample_post(self, x, n_samp):
        """Get samples from gp posterior for a single input x."""
        sample_list = self.sample_post_list([x], n_samp)
        return sample_list[0]

    def sample_post_pred_list(self, x_list, n_samp, full_cov=True):
        """Get samples from gp posterior predictive for each x in x_list."""
        # For now, return posterior (assuming low-noise case)
        # TODO: update this function
        return self.sample_post_list(x_list, n_samp, full_cov)

    def sample_post_pred(self, x, n_samp):
        """Get samples from gp posterior predictive for a single input x."""
        sample_list = self.sample_post_pred_list([x], n_samp)
        return sample_list[0]

    def get_normal_samples(self, mu, cov, n_samp, full_cov):
        """Return normal samples."""
        if full_cov:
            sample_list = list(sample_mvn(mu, cov, n_samp))
        else:
            sample_list = list(
                np.random.normal(
                    mu.reshape(
                        -1,
                    ),
                    cov.reshape(
                        -1,
                    ),
                    size=(n_samp, len(mu)),
                )
            )
        x_list_sample_list = list(np.stack(sample_list).T)
        return x_list_sample_list

    def initialize_function_sample_list(self, n_samp=1):
        """Initialize a list of n_samp function samples."""
        self.fsl_queries = [Namespace(x=[], y=[]) for _ in range(n_samp)]

    def call_function_sample_list(self, x_list):
        """Call a set of posterior function samples on respective x in x_list."""
        y_list = []

        for x, query_ns in zip(x_list, self.fsl_queries):
            # Get y for a posterior function sample at x
            comb_data = self.combine_data_namespaces(self.data, query_ns)

            if x is not None:
                mu, cov = self.gp_post_wrapper([x], comb_data, True)
                y = self.get_normal_samples(mu, cov, 1, True)
                y = y[0][0]

                # Update query history
                query_ns.x.append(x)
                query_ns.y.append(y)
            else:
                y = None

            y_list.append(y)

        return y_list

    def combine_data_namespaces(self, ns1, ns2):
        """Combine two data Namespaces, with fields x and y."""
        ns = Namespace()
        ns.x = ns1.x + ns2.x
        ns.y = ns1.y + ns2.y
        return ns


class TFSimpleGp(SimpleGp):
    """
    A version of SimpleGp that replaces all the fundamental operations
    with native TensorFlow operations so as to promote differentiability
    both through function calls to the arguments and through function calls
    to the dataset.

    We also hope to modify this class to be able to be used in tf.Function
    code (not sure what all that entails yet, besides all data being passed
    around as TF tensors.
    """

    def set_kernel(self, params):
        """Set self.params.kernel. Uses the TF versions"""
        self.params.kernel_str = getattr(params, "kernel_str", "rbf")

        if self.params.kernel_str == "rbf":
            self.params.kernel = tf_kern_exp_quad_ard

        elif self.params.kernel_str == "rbf_periodic":
            # will implement this in tf when needed
            raise NotImplementedError()

    def set_data(self, data, lmat=None, smat=None):
        """
        Data should be given as a dict or Namespace where
        data.x is an n x d_x tf tensor and
        data.y is an n x 1 tf tensor
        """
        assert data is not None
        data = dict_to_namespace(data)
        # hopefully a noop if it is already tf
        data.x = tf.convert_to_tensor(data.x)
        data.y = tf.convert_to_tensor(data.y)
        self.data = data
        x_train = self.data.x
        y_train = self.data.y
        kernel = self.params.kernel
        ls = self.params.ls
        alpha = self.params.alpha
        sigma = self.params.sigma
        assert (lmat is None) == (smat is None)

        if lmat is not None:
            self.lmat = lmat
            self.smat = smat
        else:
            k11_nonoise = kernel(x_train, x_train, ls, alpha)
            self.lmat = tf_get_cholesky_decomp(k11_nonoise, sigma, "try_first")
            self.smat = tf_solve_upper_triangular(
                tf.transpose(self.lmat), tf_solve_lower_triangular(self.lmat, y_train)
            )
        # if self.smat.ndim == 1:
        #     self.smat = self.smat[None, :]

    def get_prior_mu_cov(self, x_list, full_cov=True):
        raise NotImplementedError()

    def get_post_mu_cov(self, x_list, full_cov=True):
        """
        Return GP posterior parameters: mean (mu) and covariance (cov). If there is no
        data, return the GP prior parameters.

        Parameters
        ----------
        x_list : list
            List of numpy ndarrays, each representing a domain point.
        full_cov : bool
            If True, return covariance matrix. If False, return list of standard
            deviations.

        Returns
        -------
        mu : ndarray
            A numpy 1d ndarray with len=len(x_list) of floats, corresponding to
            posterior mean for each x in x_list.
        cov : ndarray
            If full_cov is False, return a numpy 1d ndarray with len=len(x_list) of
            floats, corresponding to posterior standard deviations for each x in x_list.
            If full_cov is True, return the covariance matrix as a numpy ndarray
            (len(x_list) x len(x_list)).
        """
        k21 = self.params.kernel(x_list, self.data.x, self.params.ls, self.params.alpha)
        mu2 = tf.matmul(k21, self.smat)

        k22 = self.params.kernel(x_list, x_list, self.params.ls, self.params.alpha)
        vmat = tf_solve_lower_triangular(self.lmat, tf.transpose(k21))
        k2 = k22 - tf.matmul(tf.transpose(vmat), vmat)
        if full_cov is False:
            k2 = tf.sqrt(tf.linalg.diag_part(k2))

        # Return mean and cov matrix (or std-dev array if full_cov=False)
        mu = tf.squeeze(mu2)
        return mu, k2

    def gp_post_wrapper(self, x_list, data, full_cov=True):
        raise NotImplementedError()

    def get_post_mu_cov_single(self, x):
        raise NotImplementedError()

    def sample_prior_list(self, x_list, n_samp, full_cov=True):
        raise NotImplementedError()

    def sample_prior(self, x, n_samp):
        raise NotImplementedError()

    def sample_post_list(self, x_list, n_samp, full_cov=True):
        raise NotImplementedError()

    def sample_post(self, x, n_samp):
        raise NotImplementedError()

    def sample_post_pred_list(self, x_list, n_samp, full_cov=True):
        raise NotImplementedError()

    def sample_post_pred(self, x, n_samp):
        raise NotImplementedError()

    def get_normal_samples(self, mu, cov, n_samp, full_cov):
        raise NotImplementedError()

    def combine_data_namespaces(self, ns1, ns2):
        raise NotImplementedError()
