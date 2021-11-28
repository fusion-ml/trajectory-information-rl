"""
Code for Gaussian processes using GPflow and GPflowSampling.
"""

from argparse import Namespace
import copy
import numpy as np
import tensorflow as tf
from gpflow import kernels
from gpflow.config import default_float as floatx

from gpflow_sampling.models import PathwiseGPR
from .gpfs.periodic import Periodic
from .simple_gp import SimpleGp
from ..util.base import Base
from ..util.misc_util import dict_to_namespace, suppress_stdout_stderr
from ..util.domain_util import unif_random_sample_domain


class GpfsGp(SimpleGp):
    """
    GP model using GPFlowSampling.
    """

    def set_params(self, params):
        """Set self.params, the parameters for this model."""
        super().set_params(params)
        params = dict_to_namespace(params)

        # Set self.params
        self.params.name = getattr(params, 'name', 'GpfsGp')
        self.params.n_bases = getattr(params, 'n_bases', 1000)
        self.params.n_dimx = getattr(params, 'n_dimx', 1)
        self.set_kernel(params)

    def set_kernel(self, params):
        """Set GPflow kernel."""
        super().set_kernel(params)

        if self.params.kernel_str == 'rbf':
            gpf_kernel = kernels.SquaredExponential(
                variance=self.params.alpha**2, lengthscales=self.params.ls
            )

        elif self.params.kernel_str == 'rbf_periodic':
            period = params.period

            per_dims = params.periodic_dims
            per_dims_ls_idx = per_dims[0] if len(per_dims)==1 else list(per_dims)
            rbf_dims = [i for i in range(self.params.n_dimx) if i not in per_dims]
            rbf_dims_ls_idx = rbf_dims[0] if len(rbf_dims)==1 else list(rbf_dims)

            gpf_kernel_1 = kernels.SquaredExponential(
                variance=self.params.alpha**2,
                lengthscales=self.params.ls[rbf_dims_ls_idx],
                active_dims=rbf_dims,
            )
            gpf_kernel_2 = kernels.SquaredExponential(
                variance=1.0,
                lengthscales=self.params.ls[per_dims_ls_idx],
                active_dims=per_dims,
            )
            gpf_kernel_per = kernels.Periodic(gpf_kernel_2, period=period)
            gpf_kernel = kernels.Product([gpf_kernel_1, gpf_kernel_per])

        self.params.gpf_kernel = gpf_kernel

    def set_data(self, data):
        """Set self.data."""
        super().set_data(data)
        self.tf_data = Namespace()
        self.tf_data.x = tf.convert_to_tensor(np.array(self.data.x))
        self.tf_data.y = tf.convert_to_tensor(
            np.array(self.data.y).reshape(-1, 1)
        )
        self.set_model()

    def set_model(self):
        """Set GPFlowSampling as self.params.model."""
        self.params.model = PathwiseGPR(
            data=(self.tf_data.x, self.tf_data.y),
            kernel=self.params.gpf_kernel,
            noise_variance=self.params.sigma**2,
        )

    def initialize_function_sample_list(self, n_fsamp=1):
        """Initialize a list of n_fsamp function samples."""
        n_bases = self.params.n_bases
        paths = self.params.model.generate_paths(num_samples=n_fsamp, num_bases=n_bases)
        _ = self.params.model.set_paths(paths)

        Xinit = tf.random.uniform(
            [n_fsamp, self.params.n_dimx], minval=0.0, maxval=0.1, dtype=floatx()
        )
        self.fsl_xvars = Xinit.numpy() #### TODO initialize directly with numpy
        self.n_fsamp = n_fsamp

    @tf.function
    def call_fsl_on_xvars(self, model, xvars, sample_axis=0):
        """Call fsl on fsl_xvars."""
        fvals = model.predict_f_samples(Xnew=xvars, sample_axis=sample_axis)
        return fvals

    def call_function_sample_list(self, x_list):
        """Call a set of posterior function samples on respective x in x_list."""

        # Replace Nones in x_list with first non-None value
        x_list = self.replace_x_list_none(x_list)

        # Set fsl_xvars as x_list, call fsl, return y_list
        self.fsl_xvars = np.array(x_list)

        y_tf = self.call_fsl_on_xvars(self.params.model, self.fsl_xvars)
        y_list = list(y_tf.numpy().reshape(-1))
        return y_list

    def call_function_sample_list_mean(self, x):
        """
        Call a set of posterior function samples on an input x and return mean of
        outputs.
        """

        # Construct x_dupe_list
        x_dupe_list = [x for _ in range(self.n_fsamp)]

        # Set fsl_xvars as x_dupe_list, call fsl, return y_list
        self.fsl_xvars = np.array(x_dupe_list)
        y_tf = self.call_fsl_on_xvars(self.params.model, self.fsl_xvars)
        y_mean = y_tf.numpy().reshape(-1).mean()
        return y_mean

    def replace_x_list_none(self, x_list):
        """Replace any Nones in x_list with first non-None value and return x_list."""

        # Set new_val as first non-None element of x_list
        new_val = next(x for x in x_list if x is not None)

        # Replace all Nones in x_list with new_val
        x_list_new = [new_val if x is None else x for x in x_list]

        return x_list_new


class MultiGpfsGp(Base):
    """
    Simple multi-output GP model using GPFlowSampling. To do this, this class duplicates
    the model in GpfsGp multiple times (and uses same kernel and other parameters in
    each duplication).
    """

    def __init__(self, params=None, data=None, verbose=True):
        super().__init__(params, verbose)
        self.set_data(data)
        self.set_gpfsgp_list()

    def set_params(self, params):
        """Set self.params, the parameters for this model."""
        super().set_params(params)
        params = dict_to_namespace(params)

        self.params.name = getattr(params, 'name', 'MultiGpfsGp')
        self.params.n_dimy = getattr(params, 'n_dimy', 1)
        self.params.gp_params = getattr(params, 'gp_params', {})

    def set_data(self, data):
        """Set self.data."""
        if data is None:
            self.data = Namespace(x=[], y=[])
        else:
            data = dict_to_namespace(data)
            self.data = copy.deepcopy(data)

    def set_gpfsgp_list(self):
        """Set self.gpfsgp_list by instantiating a list of GpfsGp objects."""
        data_list = self.get_data_list(self.data)
        gp_params_list = self.get_gp_params_list()

        # Each GpfsGp verbose set to same as self.params.verbose
        verb = self.params.verbose
        self.gpfsgp_list = [
            GpfsGp(gpp, dat, verb) for gpp, dat in zip(gp_params_list, data_list)
        ]

    def initialize_function_sample_list(self, n_samp=1):
        """
        Initialize a list of n_samp function samples, for each GP in self.gpfsgp_list.
        """
        for gpfsgp in self.gpfsgp_list:
            gpfsgp.initialize_function_sample_list(n_samp)

    def call_function_sample_list(self, x_list):
        """
        Call a set of posterior function samples on respective x in x_list, for each GP
        in self.gpfsgp_list.
        """
        y_list_list = [
            gpfsgp.call_function_sample_list(x_list) for gpfsgp in self.gpfsgp_list
        ]

        # y_list is a list, where each element is a list representing a multidim y
        y_list = [list(x) for x in zip(*y_list_list)]
        return y_list

    def call_function_sample_list_mean(self, x):
        """
        Call a set of posterior function samples on an input x and return mean of
        outputs, for each GP in self.gpfsgp_list.
        """

        # y_vec is a list of outputs for a single x (one output per gpfsgp)
        y_vec = [
            gpfsgp.call_function_sample_list_mean(x) for gpfsgp in self.gpfsgp_list
        ]
        return y_vec

    def get_post_mu_cov(self, x_list, full_cov=False):
        """Returns a list of mu, and a list of cov/std."""
        mu_list, cov_list = [], []
        for gpfsgp in self.gpfsgp_list:
            # Call usual 1d gpfsgp gp_post_wrapper
            mu, cov = gpfsgp.get_post_mu_cov(x_list, full_cov)
            mu_list.append(mu)
            cov_list.append(cov)

        return mu_list, cov_list

    def gp_post_wrapper(self, x_list, data, full_cov=True):
        """Returns a list of mu, and a list of cov/std."""

        data_list = self.get_data_list(data)
        mu_list = []
        cov_list = []

        for gpfsgp, data_single in zip(self.gpfsgp_list, data_list):
            # Call usual 1d gpfsgp gp_post_wrapper
            mu, cov = gpfsgp.gp_post_wrapper(x_list, data_single, full_cov)
            mu_list.append(mu)
            cov_list.append(cov)

        return mu_list, cov_list


    def get_data_list(self, data):
        """
        Return list of Namespaces, where each is a version of data containing only one
        of the dimensions of data.y (and the full data.x).
        """
        data_list = []
        for j in range(self.params.n_dimy):
            data_list.append(Namespace(x=data.x, y=[yi[j] for yi in data.y]))

        return data_list

    def get_gp_params_list(self):
        """
        Return list of gp_params dicts (same length as self.data_list), by parsing
        self.params.gp_params.
        """
        gp_params_list = [
            copy.deepcopy(self.params.gp_params) for _ in range(self.params.n_dimy)
        ]

        hyps = ['ls', 'alpha', 'sigma']
        for hyp in hyps:
            if not isinstance(self.params.gp_params.get(hyp, 1), (float, int)):
                # If hyp exists in dict, and is not (float, int), assume is list of hyp
                for idx, gpp in enumerate(gp_params_list):
                    gpp[hyp] = self.params.gp_params[hyp][idx]

        return gp_params_list


class BatchGpfsGp(GpfsGp):
    """
    GPFlowSampling GP model tailored to batch algorithms with BAX.
    """

    def set_params(self, params):
        """Set self.params, the parameters for this model."""
        super().set_params(params)
        params = dict_to_namespace(params)

        # Set self.params
        self.params.name = getattr(params, 'name', 'BatchGpfsGp')

    def initialize_function_sample_list(self, n_fsamp=1):
        """Initialize a list of n_fsamp function samples."""
        n_bases = self.params.n_bases
        paths = self.params.model.generate_paths(num_samples=n_fsamp, num_bases=n_bases)
        _ = self.params.model.set_paths(paths)
        self.n_fsamp = n_fsamp

    def initialize_fsl_xvars(self, n_batch):
        """
        Initialize set.fsl_xvars, a tf.Variable of correct size, given batch size
        n_batch.
        """
        Xinit = tf.zeros([self.n_fsamp, n_batch, self.params.n_dimx], dtype=floatx())
        self.fsl_xvars = Xinit.numpy() #### TODO initialize directly with numpy

    def call_function_sample_list(self, x_batch_list):
        """
        Call a set of posterior function samples on respective x_batch (a list of
        inputs) in x_batch_list.
        """

        # Replace empty x_batch and convert all x_batch to max batch size
        x_batch_list_new, max_n_batch = self.reformat_x_batch_list(x_batch_list)

        # Only re-initialize fsl_xvars if max_n_batch is larger than self.max_n_batch
        # note: I observerd that this doesn't work -- it still crashes if you pass in a batch that is smaller than
        # self.max_n_batch. So I'm changing this to !=
        if max_n_batch != getattr(self, "max_n_batch", 0):
            self.max_n_batch = max_n_batch
            self.initialize_fsl_xvars(max_n_batch)

        # Set fsl_xvars as x_batch_list_new
        self.fsl_xvars = np.array(x_batch_list_new)

        # Call fsl on fsl_xvars, return y_list
        y_tf = self.call_fsl_on_xvars(self.params.model, self.fsl_xvars)

        # Return list of y_batch lists, each cropped to same size as original x_batch
        y_batch_list = []
        for yarr, x_batch in zip(y_tf.numpy(), x_batch_list):
            y_batch = list(yarr.reshape(-1))[:len(x_batch)]
            y_batch_list.append(y_batch)

        return y_batch_list

    def reformat_x_batch_list(self, x_batch_list):
        """Make all batches the same size and replace all empty lists."""

        # Find first non-empty list and use first entry as dummy value
        dum_val = next(x_batch for x_batch in x_batch_list if len(x_batch) > 0)[0]
        max_n_batch = max(len(x_batch) for x_batch in x_batch_list)

        # duplicate and reformat each x_batch in x_batch_list, add to x_batch_list_new
        x_batch_list_new = []
        for x_batch in x_batch_list:
            x_batch_dup = [*x_batch]
            x_batch_dup.extend([dum_val] * (max_n_batch - len(x_batch_dup)))
            x_batch_list_new.append(x_batch_dup)

        return x_batch_list_new, max_n_batch


class BatchMultiGpfsGp(MultiGpfsGp):
    """
    Batch version of MultiGpfsGp model, which is tailored to batch algorithms with BAX.
    To do this, this class duplicates the model in BatchGpfsGp multiple times.
    """

    def set_params(self, params):
        """Set self.params, the parameters for this model."""
        super().set_params(params)
        params = dict_to_namespace(params)

        self.params.name = getattr(params, 'name', 'MultiBatchGpfsGp')

    def set_gpfsgp_list(self):
        """Set self.gpfsgp_list by instantiating a list of BatchGpfsGp objects."""
        data_list = self.get_data_list(self.data)
        gp_params_list = self.get_gp_params_list()

        # Each BatchGpfsGp verbose set to same as self.params.verbose
        verb = self.params.verbose
        self.gpfsgp_list = [
            BatchGpfsGp(gpp, dat, verb) for gpp, dat in zip(gp_params_list, data_list)
        ]

    def call_function_sample_list(self, x_batch_list):
        """
        Call a set of posterior function samples on respective x in x_list, for each GP
        in self.gpfsgp_list.
        """
        y_batch_list_list = [
            gpfsgp.call_function_sample_list(x_batch_list) for gpfsgp in self.gpfsgp_list
        ]

        # We define y_batch_multi_list to be a list, where each element is: a list of
        # multi-output-y (one per n_batch)
        y_batch_multi_list = [list(zip(*ybl)) for ybl in zip(*y_batch_list_list)] # ugly

        # Convert from list of lists of tuples to all lists
        y_batch_multi_list = [[list(tup) for tup in li] for li in y_batch_multi_list]

        return y_batch_multi_list

    def call_function_sample_list_mean(self, x):
        """
        Call a set of posterior function samples on an input x and return mean of
        outputs, for each GP in self.gpfsgp_list.
        """
        # TODO: possibly implement for BatchMultiGpfsGp for sample approximation of
        # posterior mean
        pass

    def sample_post_list(self, x_list, n_samp, full_cov=False):
        '''
        This is going to return a triply-nested list of shape
        len(x_list) x n_samp x len(self.gpfsgs_list)
        '''
        assert len(self.data.x) > 0
        mu_list, cov_list = self.get_post_mu_cov(x_list, full_cov)
        return self.get_normal_samples(mu_list, cov_list, n_samp, full_cov)

    def get_normal_samples(self, mu_list, cov_list, n_samp, full_cov):
        # TODO: I don't think this will work for the multivariate case
        # copied from simpleGP but I think this code won't work.
        # this is here as an initial attempt
        if full_cov:
            sample_list = list(sample_mvn(mu, cov, n_samp))
        else:
            sample_list = list(
                np.random.normal(
                    mu.reshape(-1,), cov.reshape(-1,), size=(n_samp, len(mu))
                )
            )
        x_list_sample_list = list(np.stack(sample_list).T)
        return x_list_sample_list
