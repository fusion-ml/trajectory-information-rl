"""
Acquisition functions.
"""

from argparse import Namespace
import copy
import numpy as np
import tensorflow as tf
import jax
import jax.numpy as jnp
import logging
from collections import defaultdict
from scipy.stats import norm as sps_norm
from functools import partial
from tqdm import tqdm, trange
import math

from ..util.base import Base
from ..util.misc_util import dict_to_namespace, flatten
from ..util.timing import Timer
from ..models.function import FunctionSample
from ..models.gp.jax_gp_utils import (
    get_lmats_smats,
    get_pred_covs,
    construct_jax_kernels,
)
from ..alg.algorithms import AlgorithmSet, BatchAlgorithmSet


class AcqFunction(Base):
    """
    Class for computing acquisition functions.
    """

    def __init__(self, params=None, model=None, verbose=True, **kwargs):
        """
        Parameters
        ----------
        params : Namespace_or_dict
            Namespace or dict of parameters for the AcqFunction.
        model : SimpleGp
            Instance of a SimpleGp or child class.
        verbose : bool
            If True, print description string.
        """
        super().__init__(params, verbose)
        self.set_model(model)

    def set_params(self, params):
        """Set self.params, the parameters for the AcqFunction."""
        super().set_params(params)
        params = dict_to_namespace(params)
        self.params.name = getattr(params, "name", "AcqFunction")

    def set_model(self, model):
        """Set self.model, the model underlying the acquisition function."""
        if not model:
            raise ValueError("The model input parameter cannot be None.")
        else:
            self.model = copy.deepcopy(model)

    def initialize(self):
        """Initialize the acquisition function before acquisition optimization."""
        pass

    def __call__(self, x_list):
        """Class is callable and returns acquisition function on x_list."""

        # Random acquisition function
        acq_list = [np.random.random() for x in x_list]

        return acq_list


class RandAcqFunction(AcqFunction):
    """
    Class for random search acquisition functions.
    """

    def set_params(self, params):
        """Set self.params, the parameters for the AcqFunction."""
        super().set_params(params)

        params = dict_to_namespace(params)
        self.params.name = getattr(params, "name", "RandAcqFunction")

    def __call__(self, x_list):
        """Class is callable and returns acquisition function on x_list."""

        # Random acquisition function
        acq_list = [np.random.random() for x in x_list]

        return acq_list


class AlgoAcqFunction(AcqFunction):
    """
    Class for computing acquisition functions involving algorithms, such as entropy
    search and BAX methods.
    """

    def __init__(self, params=None, model=None, algorithm=None, verbose=True):
        """
        Parameters
        ----------
        params : Namespace_or_dict
            Namespace or dict of parameters for the AcqFunction.
        model : SimpleGp
            Instance of a SimpleGp or child.
        algorithm : Algorithm
            Instance of an Algorithm or child.
        verbose : bool
            If True, print description string.
        """
        super().__init__(params, verbose)
        self.set_model(model)
        self.set_algorithm(algorithm)

    def set_params(self, params):
        """Set self.params, the parameters for the AcqFunction."""
        super().set_params(params)

        params = dict_to_namespace(params)
        self.params.name = getattr(params, "name", "AlgoAcqFunction")
        self.params.n_path = getattr(params, "n_path", 100)
        self.params.crop = getattr(params, "crop", True)

    def set_algorithm(self, algorithm):
        """Set self.algorithm for this acquisition function."""
        if not algorithm:
            raise ValueError("The algorithm input parameter cannot be None.")
        else:
            self.algorithm = algorithm.get_copy()

    def initialize(self):
        """
        Initialize the acquisition function before acquisition optimization. Draw
        samples of the execution path and algorithm output from functions sampled from
        the model.
        """
        exe_path_list, output_list, full_list = self.get_exe_path_and_output_samples()

        # Set self.output_list
        self.output_list = output_list
        self.exe_path_full_list = full_list

        # Set self.exe_path_list to list of full or cropped exe paths
        if self.params.crop:
            self.exe_path_list = exe_path_list
        else:
            self.exe_path_list = full_list

    def get_exe_path_and_output_samples_loop(self):
        """
        Return exe_path_list and output_list respectively containing self.params.n_path
        exe_path samples and associated outputs, using self.model and self.algorithm.
        """
        exe_path_list = []
        output_list = []
        with Timer(f"Sample {self.params.n_path} execution paths"):
            for _ in range(self.params.n_path):
                fs = FunctionSample(verbose=False)
                fs.set_model(self.model)
                exe_path, output = self.algorithm.run_algorithm_on_f(fs)
                exe_path_list.append(exe_path)
                output_list.append(output)

        return exe_path_list, output_list

    def get_exe_path_and_output_samples(self):
        """
        Return exe_path_list and output_list respectively containing self.params.n_path
        exe_path samples and associated outputs, using self.model and self.algorithm.
        """
        exe_path_list = []
        output_list = []
        with Timer(f"Sample {self.params.n_path} execution paths"):
            # Initialize model fsl
            self.model.initialize_function_sample_list(self.params.n_path)

            if getattr(self.algorithm.params, "is_batch", False):
                algoset = BatchAlgorithmSet(self.algorithm)
            else:
                algoset = AlgorithmSet(self.algorithm)

            # Run algorithm on function sample list
            f_list = self.model.call_function_sample_list
            exe_path_full_list, output_list = algoset.run_algorithm_on_f_list(
                f_list, self.params.n_path
            )

            # Get crop of each exe_path in exe_path_list
            exe_path_list = algoset.get_exe_path_list_crop()

        return exe_path_list, output_list, exe_path_full_list


class BaxAcqFunction(AlgoAcqFunction):
    """
    Class for computing BAX acquisition functions.
    """

    def set_params(self, params):
        """Set self.params, the parameters for the AcqFunction."""
        super().set_params(params)

        params = dict_to_namespace(params)
        self.params.name = getattr(params, "name", "BaxAcqFunction")
        self.params.acq_str = getattr(params, "acq_str", "exe")
        self.params.min_neighbors = getattr(params, "min_neighbors", 10)
        self.params.max_neighbors = getattr(params, "max_neighbors", 30)
        self.params.dist_thresh = getattr(params, "dist_thresh", 1.0)
        self.params.dist_thresh_init = getattr(params, "dist_thresh_init", 20.0)
        self.params.dist_thresh_inc = getattr(params, "dist_thresh_inc", 0.5)
        self.params.min_n_clust = getattr(params, "min_n_clust", 5)

    def entropy_given_normal_std(self, std_arr):
        """Return entropy given an array of 1D normal standard deviations."""
        entropy = np.log(std_arr) + np.log(np.sqrt(2 * np.pi)) + 0.5
        return entropy

    def acq_exe_normal(self, post_std, samp_std_list):
        """
        Execution-path-based acquisition function: EIG on the execution path, via
        predictive entropy, for normal posterior predictive distributions.
        """

        # Compute entropies for posterior predictive
        h_post = self.entropy_given_normal_std(post_std)

        # Compute entropies for posterior predictive given execution path samples
        h_samp_list = []
        for samp_std in samp_std_list:
            h_samp = self.entropy_given_normal_std(samp_std)
            h_samp_list.append(h_samp)

        avg_h_samp = np.mean(h_samp_list, 0)
        acq_exe = h_post - avg_h_samp
        return acq_exe

    def acq_out_normal(self, post_std, samp_mean_list, samp_std_list, output_list):
        """
        Algorithm-output-based acquisition function: EIG on the algorithm output, via
        predictive entropy, for normal posterior predictive distributions.
        """
        # Compute entropies for posterior predictive
        h_post = self.entropy_given_normal_std(post_std)

        # Get list of idx-list-per-cluster
        dist_thresh = self.params.dist_thresh_init
        cluster_idx_list = self.get_cluster_idx_list(output_list, dist_thresh)

        # -----
        print("\t- clust_idx_list initial details:")
        len_list = [len(clust) for clust in cluster_idx_list]
        print(
            f"\t\t- min len_list: {np.min(len_list)},  max len_list: {np.max(len_list)},  len(len_list): {len(len_list)}"
        )  # NOQA
        # -----

        # Filter clusters that are too small (looping to find optimal dist_thresh)
        min_nn = self.params.min_neighbors
        min_n_clust = self.params.min_n_clust
        min_dist_thresh = self.params.dist_thresh

        cluster_idx_list_new = [
            clust for clust in cluster_idx_list if len(clust) > min_nn
        ]
        # -----
        print("\t- clust_idx_list_NEW details:")
        len_list = [len(clust) for clust in cluster_idx_list_new]
        print(
            f"\t\t- min len_list: {np.min(len_list)},  max len_list: {np.max(len_list)},  len(len_list): {len(len_list)}"
        )
        # -----
        while (
            len(cluster_idx_list_new) > min_n_clust and dist_thresh >= min_dist_thresh
        ):
            cluster_idx_list_keep = cluster_idx_list_new
            dist_thresh -= self.params.dist_thresh_inc
            print(f"NOTE: dist_thresh = {dist_thresh}")
            cluster_idx_tmp = self.get_cluster_idx_list(output_list, dist_thresh)
            cluster_idx_list_new = [
                clust for clust in cluster_idx_tmp if len(clust) > min_nn
            ]

        try:
            cluster_idx_list = cluster_idx_list_keep
        except UnboundLocalError:
            print(
                "WARNING: cluster_idx_list_keep not assigned, using cluster_idx_list."
            )
            pass

        ## Only remove small clusters if there are enough big clusters
        # if len(cluster_idx_list_new) > self.params.min_n_clust:
        # cluster_idx_list = cluster_idx_list_new

        # -----
        len_list = [len(clust) for clust in cluster_idx_list]
        print("\t- clust_idx_list final details:")
        print(
            f"\t\t- min len_list: {np.min(len_list)},  max len_list: {np.max(len_list)},  len(len_list): {len(len_list)}"
        )
        print(f"\t\tFound dist_thresh: {dist_thresh}")
        # -----

        # Compute entropies for posterior predictive given execution path samples
        h_cluster_list = []
        std_cluster_list = []
        mean_cluster_list = []
        for idx_list in cluster_idx_list:
            # Mean of the mixture
            samp_mean_cluster_list = [samp_mean_list[idx] for idx in idx_list]
            samp_mean_cluster = np.mean(samp_mean_cluster_list, 0)
            mean_cluster_list.append(samp_mean_cluster)

            # Std of the mixture
            samp_std_cluster_list = [samp_std_list[idx] for idx in idx_list]
            smcls = [smc**2 for smc in samp_mean_cluster_list]
            sscls = [ssc**2 for ssc in samp_std_cluster_list]
            sum_smcls_sscls = [smcs + sscs for smcs, sscs in zip(smcls, sscls)]
            samp_sec_moment_cluster = np.mean(sum_smcls_sscls, 0)
            samp_var_cluster = samp_sec_moment_cluster - samp_mean_cluster**2
            samp_std_cluster = np.sqrt(samp_var_cluster)
            std_cluster_list.append(samp_std_cluster)

            # Entropy of the Gaussian approximation to the mixture
            h_cluster = self.entropy_given_normal_std(samp_std_cluster)
            h_cluster_list.extend([h_cluster])

        avg_h_cluster = np.mean(h_cluster_list, 0)
        acq_out = h_post - avg_h_cluster

        # Store variables
        self.cluster_idx_list = cluster_idx_list
        self.mean_cluster_list = mean_cluster_list
        self.std_cluster_list = std_cluster_list

        return acq_out

    def get_cluster_idx_list(self, output_list, dist_thresh):
        """
        Cluster outputs in output_list (based on nearest neighbors, and using
        dist_thresh as a nearness threshold) and return list of idx-list-per-cluster.
        """

        # Build distance matrix
        dist_fn = self.algorithm.get_output_dist_fn()
        dist_mat = [[dist_fn(o1, o2) for o1 in output_list] for o2 in output_list]

        # Build idx_arr_list and dist_arr_list
        idx_arr_list = []
        dist_arr_list = []
        for row in dist_mat:
            idx_sort = np.argsort(row)
            dist_sort = np.array([row[i] for i in idx_sort])

            # Keep at most max_neighbors, as long as they are within dist_thresh
            dist_sort = dist_sort[: self.params.max_neighbors]
            row_idx_keep = np.where(dist_sort < dist_thresh)[0]

            idx_arr = idx_sort[row_idx_keep]
            idx_arr_list.append(idx_arr)

            dist_arr = dist_sort[row_idx_keep]
            dist_arr_list.append(dist_arr)

        return idx_arr_list

    def acq_is_normal(
        self, post_std, samp_mean_list, samp_std_list, output_list, x_list
    ):
        """
        Algorithm-output-based acquisition function: EIG on the algorithm output, via
        the importance sampling strategy, for normal posterior predictive distributions.
        """
        # Compute list of means and stds for full execution path
        samp_mean_list_full = []
        samp_std_list_full = []
        for exe_path in self.exe_path_full_list:
            comb_data = Namespace()
            comb_data.x = self.model.data.x + exe_path.x
            comb_data.y = self.model.data.y + exe_path.y
            samp_mean, samp_std = self.model.gp_post_wrapper(
                x_list, comb_data, full_cov=False
            )
            samp_mean_list_full.append(samp_mean)
            samp_std_list_full.append(samp_std)

        # Compute entropies for posterior predictive
        h_post = self.entropy_given_normal_std(post_std)

        # Get list of idx-list-per-cluster
        cluster_idx_list = self.get_cluster_idx_list(
            output_list, self.params.dist_thresh
        )

        # -----
        print("\t- clust_idx_list details:")
        len_list = [len(clust) for clust in cluster_idx_list]
        print(
            f"\t- min len_list: {np.min(len_list)},  max len_list: {np.max(len_list)},  len(len_list): {len(len_list)}"
        )
        # -----

        ## Remove clusters that are too small
        # min_nn = self.params.min_neighbors
        # cluster_idx_list = [clust for clust in cluster_idx_list if len(clust) > min_nn]

        # -----
        len_list = [len(clust) for clust in cluster_idx_list]
        print(
            f"\t- min len_list: {np.min(len_list)},  max len_list: {np.max(len_list)},  len(len_list): {len(len_list)}"
        )
        # -----

        # Compute entropies for posterior predictive given execution path samples
        h_samp_list = []
        # for samp_mean, samp_std in zip(samp_mean_list, samp_std_list):
        for exe_idx in range(len(samp_mean_list)):
            # Unpack
            samp_mean = samp_mean_list[exe_idx]
            samp_mean_full = samp_mean_list_full[exe_idx]
            samp_std = samp_std_list[exe_idx]
            samp_std_full = samp_std_list_full[exe_idx]
            clust_idx = cluster_idx_list[exe_idx]

            # Sample from proposal distribution
            n_samp = 200
            pow_fac = 0.001
            samp_mat = np.random.normal(samp_mean, samp_std, (n_samp, len(samp_mean)))

            # Compute importance weights denominators
            mean_mat = np.vstack([samp_mean for _ in range(n_samp)])
            std_mat = np.vstack([samp_std for _ in range(n_samp)])
            pdf_mat = sps_norm.pdf(samp_mat, mean_mat, std_mat)
            # weight_mat_den = np.ones(pdf_mat.shape)
            weight_mat_den = pdf_mat

            # Compute importance weights numerators
            pdf_mat_sum = np.zeros(pdf_mat.shape)
            for idx in clust_idx:
                samp_mean_full = samp_mean_list_full[idx]
                samp_std_full = samp_std_list_full[idx]

                mean_mat = np.vstack([samp_mean_full for _ in range(n_samp)])
                std_mat = np.vstack([samp_std_full for _ in range(n_samp)])
                pdf_mat = sps_norm.pdf(samp_mat, mean_mat, std_mat)
                pdf_mat_sum = pdf_mat_sum + pdf_mat

            # weight_mat_num = np.ones(pdf_mat_sum.shape)
            weight_mat_num = pdf_mat_sum / len(clust_idx)
            weight_mat_num = weight_mat_num

            # Compute and normalize importance weights
            weight_mat = (weight_mat_num + 1e-50) / (weight_mat_den + 1.1e-50)
            weight_mat_norm = weight_mat / np.sum(weight_mat, 0)
            weight_mat_norm = (n_samp * weight_mat_norm) ** pow_fac

            # Reweight samples and compute statistics
            weight_samp = samp_mat * weight_mat_norm * n_samp
            is_mean = np.mean(weight_samp, 0)
            is_std = np.std(weight_samp, 0)
            h_samp = self.entropy_given_normal_std(is_std)
            h_samp_list.append(h_samp)

        avg_h_samp = np.mean(h_samp_list, 0)
        acq_is = h_post - avg_h_samp
        return acq_is

    def get_acq_list_batch(self, x_list):
        """Return acquisition function for a batch of inputs x_list."""

        with Timer(
            f"Compute acquisition function for a batch of {len(x_list)} points",
            level=logging.DEBUG,
        ):
            # Compute posterior, and post given each execution path sample, for x_list
            mu, std = self.model.get_post_mu_cov(x_list, full_cov=False)

            # Compute mean and std arrays for posterior given execution path samples
            mu_list = []
            std_list = []
            for exe_path in self.exe_path_list:
                comb_data = Namespace()
                comb_data.x = self.model.data.x + exe_path.x
                comb_data.y = self.model.data.y + exe_path.y
                samp_mu, samp_std = self.model.gp_post_wrapper(
                    x_list, comb_data, full_cov=False
                )
                mu_list.append(samp_mu)
                std_list.append(samp_std)

            # Compute acq_list, the acqfunction value for each x in x_list
            if self.params.acq_str == "exe":
                acq_list = self.acq_exe_normal(std, std_list)
            elif self.params.acq_str == "out":
                acq_list = self.acq_out_normal(std, mu_list, std_list, self.output_list)
            elif self.params.acq_str == "is":
                acq_list = self.acq_is_normal(
                    std, mu_list, std_list, self.output_list, x_list
                )

        # Package and store acq_vars
        self.acq_vars = {
            "mu": mu,
            "std": std,
            "mu_list": mu_list,
            "std_list": std_list,
            "acq_list": acq_list,
        }

        # Return list of acquisition function on x in x_list
        return acq_list

    def __call__(self, x_list):
        """Class is callable and returns acquisition function on x_list."""
        acq_list = self.get_acq_list_batch(x_list)
        return acq_list


class RandBaxAcqFunction(BaxAcqFunction):
    """
    Wrapper on BaxAcqFunction for random search acquisition, when we still want various
    BaxAcqFunction variables for visualizations.
    """

    def set_params(self, params):
        """Set self.params, the parameters for the AcqFunction."""
        super().set_params(params)

        params = dict_to_namespace(params)
        self.params.name = getattr(params, "name", "RandBaxAcqFunction")

    def __call__(self, x_list):
        """Class is callable and returns acquisition function on x_list."""
        acq_list = super().__call__(x_list)  # NOTE: would super()(x_list) work?
        acq_list = [np.random.uniform() for _ in acq_list]
        return acq_list


class UsBaxAcqFunction(BaxAcqFunction):
    """
    Wrapper on BaxAcqFunction for uncertainty sampling acquisition, when we still want
    various BaxAcqFunction variables for visualizations.
    """

    def set_params(self, params):
        """Set self.params, the parameters for the AcqFunction."""
        super().set_params(params)

        params = dict_to_namespace(params)
        self.params.name = getattr(params, "name", "UsBaxAcqFunction")

    def __call__(self, x_list):
        """Class is callable and returns acquisition function on x_list."""
        super().__call__(x_list)  # NOTE: would super()(x_list) work?
        acq_list = self.acq_vars["std"]
        return acq_list


class MultiBaxAcqFunction(AlgoAcqFunction):
    """
    Class for computing BAX acquisition functions.
    """

    def set_params(self, params):
        """Set self.params, the parameters for the AcqFunction."""
        super().set_params(params)

        params = dict_to_namespace(params)
        self.params.name = getattr(params, "name", "MultiBaxAcqFunction")

    def entropy_given_normal_std(self, std_arr):
        """Return entropy given an array of 1D normal standard deviations."""
        entropy = np.log(std_arr) + np.log(np.sqrt(2 * np.pi)) + 0.5
        return entropy

    def entropy_given_normal_std_list(self, std_arr_list):
        """
        Return entropy given a list of arrays, where each is an array of 1D normal
        standard deviations.
        """
        entropy_list = [
            self.entropy_given_normal_std(std_arr) for std_arr in std_arr_list
        ]
        entropy = np.sum(entropy_list, 0)
        return entropy

    def acq_exe_normal(self, post_stds, samp_stds_list):
        """
        Execution-path-based acquisition function: EIG on the execution path, via
        predictive entropy, for normal posterior predictive distributions.
        """

        # Compute entropies for posterior predictive
        h_post = self.entropy_given_normal_std_list(post_stds)

        # Compute entropies for posterior predictive given execution path samples
        h_samp_list = []
        for samp_stds in samp_stds_list:
            h_samp = self.entropy_given_normal_std_list(samp_stds)
            h_samp_list.append(h_samp)

        avg_h_samp = np.mean(h_samp_list, 0)
        acq_exe = h_post - avg_h_samp
        return acq_exe

    def get_acq_list_batch(self, x_list):
        """Return acquisition function for a batch of inputs x_list."""

        # Compute posterior, and post given each execution path sample, for x_list
        with Timer(
            f"Compute acquisition function for a batch of {len(x_list)} points",
            level=logging.DEBUG,
        ):
            # NOTE: self.model is multimodel so the following returns a list of mus and
            # a list of stds
            mus, stds = self.model.get_post_mu_cov(x_list, full_cov=False)
            assert isinstance(mus, list)
            assert isinstance(stds, list)

            # Compute mean and std arrays for posterior given execution path samples
            mus_list = []
            stds_list = []
            for exe_path in self.exe_path_list:
                comb_data = Namespace()
                comb_data.x = self.model.data.x + exe_path.x
                comb_data.y = self.model.data.y + exe_path.y

                # NOTE: self.model is multimodel so the following returns a list of mus
                # and a list of stds
                samp_mus, samp_stds = self.model.gp_post_wrapper(
                    x_list, comb_data, full_cov=False
                )
                mus_list.append(samp_mus)
                stds_list.append(samp_stds)

            # Compute acq_list, the acqfunction value for each x in x_list
            acq_list = self.acq_exe_normal(stds, stds_list)

        # Package and store acq_vars
        self.acq_vars = {
            "mus": mus,
            "stds": stds,
            "mus_list": mus_list,
            "stds_list": stds_list,
            "acq_list": acq_list,
        }

        # Return list of acquisition function on x in x_list
        return acq_list

    def __call__(self, x_list):
        """Class is callable and returns acquisition function on x_list."""
        acq_list = self.get_acq_list_batch(x_list)
        return acq_list


class SumSetBaxAcqFunction(MultiBaxAcqFunction):
    BATCH_SIZE = 100

    def set_params(self, params):
        super().set_params(params)

    def __call__(self, x_set_list):
        flat_set_list = []
        for lil_list in x_set_list:
            flat_set_list.extend(lil_list)
        acq_list = []
        nbatches = math.ceil(len(flat_set_list) / self.BATCH_SIZE)
        for i in trange(nbatches):
            acq_batch = flat_set_list[i * self.BATCH_SIZE : (i + 1) * self.BATCH_SIZE]
            acq_list.extend(self.get_acq_list_batch(acq_batch))
        current_idx = 0
        set_acq_val = []
        for lil_list in x_set_list:
            new_idx = current_idx + len(lil_list)
            set_total_acq = sum(acq_list[current_idx:new_idx])
            set_acq_val.append(set_total_acq)
            current_idx = new_idx
        return set_acq_val


class RewardSetAcqFunction(AcqFunction):
    def set_params(self, params):
        super().set_params(params)
        params = dict_to_namespace(params)
        self.params.reward_fn = params.reward_fn
        self.params.obs_dim = params.obs_dim
        self.params.action_dim = params.action_dim

    def __call__(self, x_set_list):
        """
        x_set_list should be a triply-nested list,
        where the first list is a batch, the second list is a trajectory
        (this function assumes it is sequential for reward computation)
        and the third list (or numpy array) is the actual query points of
        dimension obs_dim + action_dim
        """
        batch_size = len(x_set_list)
        x_data = np.array(x_set_list)
        rew_x = x_data[:, :-1, :].reshape(
            (-1, self.params.obs_dim + self.params.action_dim)
        )
        next_obs_data = x_data[:, 1:, : self.params.obs_dim].reshape(
            (-1, self.params.obs_dim)
        )
        rewards = (
            self.params.reward_fn(rew_x, next_obs_data)
            .reshape((batch_size, -1))
            .sum(axis=1)
        )
        return rewards


class BatchUncertaintySamplingAcqFunction(AcqFunction):
    """
    Class for computing BAX acquisition functions.
    """

    def set_params(self, params):
        """Set self.params, the parameters for the AcqFunction."""
        super().set_params(params)

        params = dict_to_namespace(params)
        self.params.name = getattr(
            params, "name", "BatchUncertaintySamplingAcqFunction"
        )
        self.base_smat = None
        self.base_lmat = None
        self.smats = defaultdict(lambda: None)
        self.lmats = defaultdict(lambda: None)
        self.kernels = construct_jax_kernels(params.gp_model_params)
        sigma = params.gp_model_params["gp_params"]["sigma"]
        self.get_lmats_smats = partial(
            get_lmats_smats, kernels=self.kernels, sigma=sigma
        )
        self.jit_fast_acq = getattr(params, "jit_fast_acq", None)

    def set_model(self, model):
        """Set self.model, the model underlying the acquisition function."""
        if not model:
            raise ValueError("The model input parameter cannot be None.")
        else:
            self.model = copy.deepcopy(model)

    @staticmethod
    def fast_acq_exe_normal(post_covs):
        signs, dets = jnp.linalg.slogdet(post_covs)
        h_post = jnp.sum(dets, axis=-1)
        return h_post

    @staticmethod
    def fast_get_acq_list_batch(x_set, x_data, y_data, base_lmats, base_smats, kernels):
        """Return acquisition function for a batch of inputs x_set_list, but do it fast."""

        # Compute posterior, and post given each execution path sample, for x_list
        x = x_data
        y = y_data
        pred_cov = get_pred_covs(x, y, x_set, base_lmats, base_smats, kernels)
        # regularization, maybe
        reg = jnp.eye(pred_cov.shape[-1])[None, None, ...] * 1e-5
        reg_pred_cov = pred_cov + reg
        acq = BatchUncertaintySamplingAcqFunction.fast_acq_exe_normal(reg_pred_cov)
        return acq

    def __call__(self, x_set_list):
        """Class is callable and returns acquisition function on x_set_list."""
        x_set_list = jnp.array(x_set_list)
        if self.jit_fast_acq is None:
            x_data = jnp.array(self.model.data.x)
            y_data = jnp.array(self.model.data.y)
            base_lmat, base_smat = self.get_lmats_smats(x_data, y_data)
            self.jit_fast_acq = jax.vmap(
                jax.jit(
                    partial(
                        self.fast_get_acq_list_batch,
                        x_data=x_data,
                        y_data=y_data,
                        base_lmats=base_lmat,
                        base_smats=base_smat,
                        kernels=self.kernels,
                    )
                )
            )
        with Timer(
            f"Compute acquisition function for a batch of {x_set_list.shape[0]} points",
            level=logging.DEBUG,
        ):
            fast_acq_list = self.jit_fast_acq(x_set_list)
        not_finites = ~jnp.isfinite(fast_acq_list)
        num_not_finite = jnp.sum(not_finites)
        if num_not_finite > 0:
            logging.warning(f"{num_not_finite} acq function results were not finite.")
            fast_acq_list = fast_acq_list.at[not_finites].set(-np.inf)
        return list(fast_acq_list)


class JointSetBaxAcqFunction(AlgoAcqFunction):
    """
    Class for computing BAX acquisition functions.
    """

    def set_params(self, params):
        """Set self.params, the parameters for the AcqFunction."""
        super().set_params(params)

        params = dict_to_namespace(params)
        self.params.name = getattr(params, "name", "JointSetBaxAcqFunction")
        self.base_smat = None
        self.base_lmat = None
        self.smats = defaultdict(lambda: None)
        self.lmats = defaultdict(lambda: None)
        self.kernels = construct_jax_kernels(params.gp_model_params)
        sigma = params.gp_model_params["gp_params"]["sigma"]
        self.get_lmats_smats = partial(
            get_lmats_smats, kernels=self.kernels, sigma=sigma
        )
        self.jit_fast_acq = getattr(params, "jit_fast_acq", None)

    def set_model(self, model):
        """Set self.model, the model underlying the acquisition function."""
        if not model:
            raise ValueError("The model input parameter cannot be None.")
        else:
            self.model = copy.deepcopy(model)
            self.conditioning_model = copy.deepcopy(model)

    @staticmethod
    def acq_exe_normal(post_covs, samp_covs_list):
        """
        Since everything about this acquisition function is the same except for the log det of the covariance, it is simply much easier to do that
        """

        # Compute entropies for posterior predictive
        signs, dets = np.linalg.slogdet(post_covs)
        h_post = np.sum(dets, axis=-1)
        # Compute entropies for posterior predictive given execution path samples
        signs, dets = np.linalg.slogdet(samp_covs_list)
        h_samp = np.sum(dets, axis=-1)
        avg_h_samp = np.mean(h_samp, axis=-1)
        acq_exe = h_post - avg_h_samp
        return acq_exe

    @staticmethod
    def fast_acq_exe_normal(post_covs, samp_covs_list):
        signs, dets = jnp.linalg.slogdet(post_covs)
        h_post = jnp.sum(dets, axis=-1)
        signs, dets = jnp.linalg.slogdet(samp_covs_list)
        h_samp = jnp.sum(dets, axis=-1)
        avg_h_samp = jnp.mean(h_samp, axis=-1)
        acq_exe = h_post - avg_h_samp
        return acq_exe

    @staticmethod
    def fast_get_acq_list_batch(
        x_set,
        x_data,
        y_data,
        base_lmats,
        base_smats,
        exe_path_list,
        kernels,
        lmats,
        smats,
    ):
        """Return acquisition function for a batch of inputs x_set_list, but do it fast."""

        # Compute posterior, and post given each execution path sample, for x_list
        x = x_data
        y = y_data
        pred_cov = get_pred_covs(x, y, x_set, base_lmats, base_smats, kernels)
        samp_cov_list = []
        for i, exe_path in enumerate(exe_path_list):
            x = jnp.concatenate([x_data, exe_path[0]], axis=0)
            y = jnp.concatenate([y_data, exe_path[1]], axis=0)
            samp_covs = get_pred_covs(x, y, x_set, lmats[i], smats[i], kernels)
            samp_cov_list.append(samp_covs)

        samp_cov_list = jnp.stack(samp_cov_list)
        # regularization, maybe
        reg = jnp.eye(samp_cov_list.shape[-1])[None, None, ...] * 1e-5
        reg_pred_cov = pred_cov + reg
        reg_samp_cov_list = samp_cov_list + reg
        acq = JointSetBaxAcqFunction.fast_acq_exe_normal(
            reg_pred_cov, reg_samp_cov_list
        )
        return acq

    def get_acq_list_batch(self, x_list):
        """Return acquisition function for a batch of inputs x_set_list."""

        # Compute posterior, and post given each execution path sample, for x_list
        with Timer(
            f"Compute acquisition function for a batch of {len(x_list)} points",
            level=logging.DEBUG,
        ):
            # NOTE: self.model is multimodel so the following returns a list of mus and
            # a list of stds
            # going to implement this with a loop first, maybe we can make it more efficient later
            acq_list = []
            for x_set in tqdm(x_list):
                mus, stds = self.model.get_post_mu_cov(x_set, full_cov=True)
                assert isinstance(mus, list)
                assert isinstance(stds, list)

                # Compute mean and std arrays for posterior given execution path samples
                mus_list = []
                stds_list = []
                for i, exe_path in enumerate(self.exe_path_list):
                    comb_data = Namespace()
                    comb_data.x = self.model.data.x + exe_path.x
                    comb_data.y = self.model.data.y + exe_path.y
                    self.conditioning_model.set_data(
                        comb_data, lmats=self.lmats[i], smats=self.smats[i]
                    )
                    self.lmats[i] = self.conditioning_model.lmats
                    self.smats[i] = self.conditioning_model.smats

                    # NOTE: self.model is multimodel so the following returns a list of mus
                    # and a list of stds
                    samp_mus, samp_stds = self.conditioning_model.get_post_mu_cov(
                        x_set,
                        full_cov=True,
                    )
                    mus_list.append(samp_mus)
                    stds_list.append(samp_stds)

                # Compute acq_list, the acqfunction value for each x in x_list
                # jax_stds = self.jax_input_list[j][0]
                # jax_samp_stds = self.jax_input_list[j][0]
                acq_list.append(self.acq_exe_normal(stds, stds_list))

        # Package and store acq_vars
        self.acq_vars = {
            "mus": mus,
            "stds": stds,
            "mus_list": mus_list,
            "stds_list": stds_list,
            "acq_list": acq_list,
        }

        # Return list of acquisition function on x in x_list
        return np.array(acq_list)

    def __call__(self, x_set_list):
        """Class is callable and returns acquisition function on x_set_list."""
        x_set_list = jnp.array(x_set_list)
        if self.jit_fast_acq is None:
            x_data = jnp.array(self.model.data.x)
            y_data = jnp.array(self.model.data.y)
            exe_paths = []
            for exe_path in self.exe_path_list:
                exe_paths.append((jnp.array(exe_path.x), jnp.array(exe_path.y)))
            lmats, smats = {}, {}
            base_lmat, base_smat = self.get_lmats_smats(x_data, y_data)
            for i, exe_path in enumerate(exe_paths):
                path_x_data = jnp.concatenate([x_data, exe_path[0]])
                path_y_data = jnp.concatenate([y_data, exe_path[1]])
                lmats[i], smats[i] = self.get_lmats_smats(path_x_data, path_y_data)
            self.jit_fast_acq = jax.vmap(
                jax.jit(
                    partial(
                        self.fast_get_acq_list_batch,
                        x_data=x_data,
                        y_data=y_data,
                        base_lmats=base_lmat,
                        base_smats=base_smat,
                        exe_path_list=exe_paths,
                        kernels=self.kernels,
                        lmats=lmats,
                        smats=smats,
                    )
                )
            )
            """
            self.jit_fast_acq = jax.vmap(partial(self.fast_get_acq_list_batch,
                                                x_data=x_data,
                                                y_data=y_data,
                                                base_lmats=base_lmat,
                                                base_smats=base_smat,
                                                exe_path_list=exe_paths,
                                                kernels=self.kernels,
                                                lmats=lmats,
                                                smats=smats))
            """
        with Timer(
            f"Compute acquisition function for a batch of {x_set_list.shape[0]} points",
            level=logging.DEBUG,
        ):
            fast_acq_list = self.jit_fast_acq(x_set_list)
        not_finites = ~jnp.isfinite(fast_acq_list)
        num_not_finite = jnp.sum(not_finites)
        if num_not_finite > 0:
            logging.warning(f"{num_not_finite} acq function results were not finite.")
            fast_acq_list = fast_acq_list.at[not_finites].set(-np.inf)
        # slow_acq_list = self.get_acq_list_batch(x_set_list)
        return list(fast_acq_list)


class MCAcqFunction(AcqFunction):
    """
    Acquisition function which wraps, duplicates, and calls a stochastic acquisition
    function multiple times, and then returns the mean.
    """

    def __init__(self, wrapped_acq_function, params):
        """Initialize with wrapped_acq_function, an AcqFunction."""

        super().__init__(params, model=None, verbose=False)
        self.num_samples_mc = self.params.num_samples_mc
        self.acq_function_copies = [
            copy.deepcopy(wrapped_acq_function) for _ in range(self.num_samples_mc)
        ]
        self.exe_path_list = []
        self.output_list = []

    def set_params(self, params):
        """Set self.params."""
        super().set_params(params)
        params = dict_to_namespace(params)

        self.params.name = "MCAcqFunction"
        self.params.num_samples_mc = params.num_samples_mc

    def set_model(self, model):
        """This acq function doesn't hold a model, just wraps other ones."""
        pass

    def initialize(self):
        """Initialize all acqfunction copies in self.acq_function_copies."""
        self.exe_path_list = []
        self.exe_path_full_list = []
        self.output_list = []
        for fn in self.acq_function_copies:
            fn.initialize()
            self.exe_path_list += fn.exe_path_list
            self.exe_path_full_list += fn.exe_path_full_list
            self.output_list += fn.output_list

    def __call__(self, x_list):
        """Call and average all acqfunctions in self.acq_function_copies."""
        lists = []
        for fn in self.acq_function_copies:
            lists.append(fn(x_list))
        return list(np.mean(lists, axis=0))

    @property
    def model(self):
        return self.acq_function_copies[0].model


class UncertaintySamplingAcqFunction(AcqFunction):
    """
    Class for computing BAX acquisition functions.
    """

    def initialize(self):
        self.exe_path_list = []
        self.output_list = []
        self.exe_path_full_list = []

    def set_params(self, params):
        """Set self.params, the parameters for the AcqFunction."""
        super().set_params(params)

        params = dict_to_namespace(params)
        self.params.name = getattr(params, "name", "UncertaintySamplingAcqFunction")
        # self.params.batch = params.batch

    def entropy_given_normal_std(self, std_arr):
        """Return entropy given an array of 1D normal standard deviations."""
        entropy = np.log(std_arr) + np.log(np.sqrt(2 * np.pi)) + 0.5
        return entropy

    def entropy_given_normal_std_list(self, std_arr_list):
        """
        Return entropy given a list of arrays, where each is an array of 1D normal
        standard deviations.
        """
        entropy_list = [
            self.entropy_given_normal_std(std_arr) for std_arr in std_arr_list
        ]
        entropy = np.sum(entropy_list, 0)
        return entropy

    def acq_exe_normal(self, post_stds):
        """
        Execution-path-based acquisition function: EIG on the execution path, via
        predictive entropy, for normal posterior predictive distributions.
        """

        # Compute entropies for posterior predictive
        h_post = self.entropy_given_normal_std_list(post_stds)
        return h_post

    def get_acq_list_batch(self, x_list):
        """Return acquisition function for a batch of inputs x_list."""

        # Compute posterior, and post given each execution path sample, for x_list
        with Timer(
            f"Compute acquisition function for a batch of {len(x_list)} points",
            level=logging.DEBUG,
        ):
            # NOTE: self.model is multimodel so the following returns a list of mus and
            # a list of stds
            mus, stds = self.model.get_post_mu_cov(x_list, full_cov=False)
            assert isinstance(mus, list)
            assert isinstance(stds, list)

            # Compute acq_list, the acqfunction value for each x in x_list
            acq_list = self.acq_exe_normal(stds)

        # Package and store acq_vars
        self.acq_vars = {
            "mus": mus,
            "stds": stds,
            "acq_list": acq_list,
        }

        # Return list of acquisition function on x in x_list
        return acq_list

    def __call__(self, x_list):
        """Class is callable and returns acquisition function on x_list."""
        acq_list = self.get_acq_list_batch(x_list)
        return acq_list


class SumSetUSAcqFunction(UncertaintySamplingAcqFunction):
    BATCH_SIZE = 100

    def set_params(self, params):
        super().set_params(params)

    def __call__(self, x_set_list):
        flat_set_list = []
        for lil_list in x_set_list:
            flat_set_list.extend(lil_list)
        acq_list = []
        nbatches = math.ceil(len(flat_set_list) / self.BATCH_SIZE)
        for i in trange(nbatches):
            acq_batch = flat_set_list[i * self.BATCH_SIZE : (i + 1) * self.BATCH_SIZE]
            acq_list.extend(self.get_acq_list_batch(acq_batch))
        current_idx = 0
        set_acq_val = []
        for lil_list in x_set_list:
            new_idx = current_idx + len(lil_list)
            set_total_acq = sum(acq_list[current_idx:new_idx])
            set_acq_val.append(set_total_acq)
            current_idx = new_idx
        return set_acq_val
