"""
Code for Gaussian processes with hyperparameter fitting/sampling using GPflow.
"""

from argparse import Namespace
import copy
import collections.abc
import numpy as np
import gpflow
import tensorflow as tf

from .simple_gp import SimpleGp
from ..util.misc_util import dict_to_namespace
from ..util.domain_util import unif_random_sample_domain


class GpflowGp(SimpleGp):
    """
    GP model using GPflow for hyperparameter fitting/sampling.
    """

    def __init__(self, params=None, data=None, verbose=True):
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
        super().__init__(params, data, verbose)
        self.set_gpflow_model()

    def set_params(self, params):
        """Set self.params, the parameters for this model."""
        super().set_params(params)
        params = dict_to_namespace(params)

        # Set self.params
        self.params.name = getattr(params, "name", "GpflowGp")
        self.params.opt_max_iter = getattr(params, "opt_max_iter", 1000)
        self.params.print_fit_hypers = getattr(params, "print_fit_hypers", False)
        self.params.fixed_mean_func = getattr(params, "fixed_mean_func", True)
        self.params.mean_func_c = getattr(params, "mean_func_c", 0.0)
        self.params.ls = getattr(params, "ls", 1.0)
        self.params.use_ard = getattr(params, "use_ard", True)
        self.params.alpha = getattr(params, "alpha", 1.0)
        self.params.fixed_noise = getattr(params, "fixed_noise", True)
        self.params.sigma = getattr(params, "sigma", 0.01)

    def set_data(self, data, lmat=None, smat=None):
        """Set self.data.

        we don't use these GPs that often, so I am not gonna implement the lmat and smat thing.
        They are no-ops for interface compatibility
        """
        if data is None:
            # Initialize self.data to be empty
            self.data = Namespace()
            self.data.x = []
            self.data.y = []
        else:
            data = dict_to_namespace(data)
            self.data = copy.deepcopy(data)

        # Set self.gpflow_data
        self.set_gpflow_data()

    def set_gpflow_data(self):
        """Set self.gpflow_data."""
        n_dimx = len(self.data.x[0])  #### NOTE: data.x must not be empty
        self.gpflow_data = (
            np.array(self.data.x).reshape(-1, n_dimx),
            np.array(self.data.y).reshape(-1, 1),
        )

    def set_gpflow_model(self):
        """Set self.model to a GPflow model."""
        # Build gpflow model on self.data
        model = self.build_new_gpflow_model_on_data(self.data)

        # Assign model to self.model
        self.model = model

    def build_new_gpflow_model_on_data(self, data):
        """Instantiate and return GPflow model on given data."""
        n_dimx = len(data.x[0])  #### NOTE: data.x must not be empty

        # Convert data to gpflow format
        gpflow_data = (
            np.array(data.x).reshape(-1, n_dimx),
            np.array(data.y).reshape(-1, 1),
        )

        # Set mean function
        mean_func = gpflow.mean_functions.Constant()
        mean_func.c.assign([self.params.mean_func_c])
        if self.params.fixed_mean_func:
            gpflow.utilities.set_trainable(mean_func.c, False)

        # Set kernel
        if self.params.use_ard:
            if isinstance(self.params.ls, collections.abc.Sequence):
                ls_init = self.params.ls
            else:
                ls_init = [self.params.ls for _ in range(n_dimx)]
        else:
            assert not isinstance(self.params.ls, collections.abc.Sequence)
            ls_init = self.params.ls
        kernel = gpflow.kernels.SquaredExponential(
            variance=self.params.alpha**2, lengthscales=ls_init
        )

        # Set GPR model
        model = gpflow.models.GPR(
            data=gpflow_data, kernel=kernel, mean_function=mean_func
        )
        model.likelihood.variance.assign(self.params.sigma**2)
        if self.params.fixed_noise:
            gpflow.utilities.set_trainable(model.likelihood.variance, False)
        return model

    def get_gpflow_model(self):
        """Return the GPflow model."""
        gpflow_model = self.model
        return gpflow_model

    def fit_hypers(self, test_data=None):
        """Fit hyperparameters."""
        opt = gpflow.optimizers.Scipy()
        opt_config = dict(maxiter=self.params.opt_max_iter)

        if test_data is None:
            # Fit hyperparameters
            if self.params.print_fit_hypers:
                print("GPflow: start hyperparameter fitting on train likelihood.")
            opt_log = opt.minimize(
                self.model.training_loss,
                self.model.trainable_variables,
                options=opt_config,
            )
            if self.params.print_fit_hypers:
                print("GPflow: end hyperparameter fitting on train likelihood.")
                gpflow.utilities.print_summary(self.model)
        else:
            # Fit hyperparameters on test likelihood
            if self.params.print_fit_hypers:
                print("GPflow: start hyperparameter fitting on test likelihood.")
            n_dimx = len(test_data.x[0])  #### NOTE: data.x must not be empty
            test_gpflow_data = (
                np.array(test_data.x).reshape(-1, n_dimx),
                np.array(test_data.y).reshape(-1, 1),
            )
            # tf.config.run_functions_eagerly(True)
            def neg_test_likelihood():
                # breakpoint()
                return -tf.reduce_sum(self.model.predict_log_density(test_gpflow_data))

            opt_log = opt.minimize(
                neg_test_likelihood, self.model.trainable_variables, options=opt_config
            )
            if self.params.print_fit_hypers:
                print("GPflow: end hyperparameter fitting on test likelihood.")
                gpflow.utilities.print_summary(self.model)
        return opt_log.fun

    def get_post_mu_cov(self, x_list, full_cov=True):
        """
        Return GP posterior parameters: mean (mu) and covariance (cov) for test points
        in x_list. If there is no data, return the GP prior parameters.

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
        mu, cov = self.get_post_mu_cov_on_model(x_list, self.model, full_cov=full_cov)
        return mu, cov

    def gp_post_wrapper(self, x_list, data, full_cov=True):
        """Wrapper for gp_post given a list of x and data Namespace."""
        if len(data.x) == 0:
            return self.get_prior_mu_cov(x_list, full_cov)

        # Build new gpflow model on data
        model = self.build_new_gpflow_model_on_data(data)

        # Compute and return mu, cov for this model
        mu, cov = self.get_post_mu_cov_on_model(x_list, model, full_cov=full_cov)
        return mu, cov

    def get_post_mu_cov_on_model(self, x_list, model, full_cov=True):
        """Return mu, cov at inputs in x_list for given gpflow model."""
        # Convert x_list inputs to correct format for gpflow
        n_dimx = len(x_list[0])
        x_arr = np.array(x_list).reshape(-1, n_dimx)

        # Get posterior parameters from gpflow model
        mu_tf, cov_tf = model.predict_f(x_arr, full_cov=full_cov)

        # Convert gpflow outputs to numpy arrays and return them
        mu = mu_tf.numpy().reshape(-1)
        if full_cov:
            cov = cov_tf.numpy()
        else:
            cov = np.sqrt(cov_tf.numpy().reshape(-1))

        return mu, cov


def get_gpflow_hypers_from_data(
    data,
    print_fit_hypers=False,
    opt_max_iter=1000,
    test_data=None,
    sigma=0.01,
    retries=5,
):
    """
    Return hypers fit by GPflow, using data Namespace (with fields x and y). Assumes y
    is a list of scalars (i.e. 1 dimensional output).
    """
    data = dict_to_namespace(data)

    data.x = np.array(data.x)
    x_dim = data.x.shape[1]
    # Fit params with GPflow on data
    passed = False
    best_nll = np.inf
    best_model = None
    for _ in range(retries):
        try:
            model_params = dict(
                print_fit_hypers=print_fit_hypers,
                opt_max_iter=opt_max_iter,
                sigma=sigma,
            )
            model_params["ls"] = list(np.random.uniform(0.0, 100, size=(x_dim,)))
            model_params["alpha"] = np.random.uniform(1.0, 20.0)
            model = GpflowGp(params=model_params, data=data)
            nll = model.fit_hypers(test_data)
            if nll < best_nll:
                best_nll = nll
                best_model = model
            passed = True
        except Exception as e:
            pass
    if not passed:
        raise ValueError("gp fitting failed")
    model = best_model
    gp_hypers = {
        "ls": model.model.kernel.lengthscales.numpy().tolist(),
        "alpha": np.sqrt(float(model.model.kernel.variance.numpy())),
        "sigma": np.sqrt(float(model.model.likelihood.variance.numpy())),
        "n_dimx": len(data.x[0]),
    }

    return gp_hypers
