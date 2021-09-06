"""
Code for Gaussian processes with hyperparameter fitting/sampling using GPflow.
"""

from argparse import Namespace
import copy
import numpy as np
import gpflow

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
        self.params.name = getattr(params, 'name', 'GpflowGp')
        self.params.opt_max_iter = getattr(params, 'opt_max_iter', 1000)
        self.params.print_fit_hypers = getattr(params, 'print_fit_hypers', False)
        self.params.fixed_mean_func = getattr(params, 'fixed_mean_func', True)
        self.params.mean_func_c = getattr(params, 'mean_func_c', 0.0)
        self.params.kernel_ls_init = getattr(params, 'kernel_ls_init', 1.0)
        self.params.kernel_var_init = getattr(params, 'kernel_var_init', 1.0)
        self.params.fixed_noise = getattr(params, 'fixed_noise', True)
        self.params.noise_var_init = getattr(params, 'noise_var_init', 0.1)

    def set_data(self, data):
        """Set self.data."""
        super().set_data(data)
        self.set_gpflow_data()

    def set_gpflow_data(self):
        """Set self.gpflow_data."""
        n_dimx = len(self.data.x[0]) #### NOTE: data.x must not be empty
        self.gpflow_data = (
            np.array(self.data.x).reshape(-1, n_dimx),
            np.array(self.data.y).reshape(-1, 1),
        )

    def set_gpflow_model(self):
        """Set self.model to a GPflow model."""
        # Set mean function
        mean_func = gpflow.mean_functions.Constant()
        mean_func.c.assign([self.params.mean_func_c])
        if self.params.fixed_mean_func:
            gpflow.utilities.set_trainable(mean_func.c, False)

        # Set kernel
        n_dimx = len(self.data.x[0]) #### NOTE: data.x must not be empty
        ls_init_list = [self.params.kernel_ls_init for _ in range(n_dimx)]
        kernel = gpflow.kernels.SquaredExponential(
            variance=self.params.kernel_var_init, lengthscales=ls_init_list
        )

        # Set GPR model
        model = gpflow.models.GPR(data=self.gpflow_data, kernel=kernel, mean_function=mean_func)
        model.likelihood.variance.assign(self.params.noise_var_init)
        if self.params.fixed_noise:
            gpflow.utilities.set_trainable(model.likelihood.variance, False)

        # Assign model to self.model
        self.model = model

    def get_gpflow_model(self):
        """Return the GPflow model."""
        gpflow_model = self.model
        return gpflow_model

    def fit_hypers(self):
        """Fit hyperparameters."""
        opt = gpflow.optimizers.Scipy()
        opt_config = dict(maxiter=self.params.opt_max_iter)

        # Fit hyperparameters
        if self.params.print_fit_hypers:
            print('GPflow: start hyperparameter fitting.')
        opt_log = opt.minimize(
            self.model.training_loss, self.model.trainable_variables, options=opt_config
        )
        if self.params.print_fit_hypers:
            print('GPflow: end hyperparameter fitting.')
            gpflow.utilities.print_summary(self.model)


def get_gpflow_hypers_from_data(data, print_fit_hypers=False):
    """
    Return hypers fit by GPflow, using data Namespace (with fields x and y). Assumes y
    is a list of scalars (i.e. 1 dimensional output).
    """
    data = dict_to_namespace(data)

    # Fit params with StanGp on data
    model_params = dict(print_fit_hypers=print_fit_hypers)
    model = GpflowGp(params=model_params, data=data)
    model.fit_hypers()
    gp_hypers = {
        'kernel_ls': model.model.kernel.lengthscales.numpy().tolist(),
        'kernel_var': float(model.model.kernel.variance.numpy()),
        'noise_var': float(model.model.likelihood.variance.numpy()),
        'n_dimx': len(data.x[0]),
    }

    return gp_hypers
