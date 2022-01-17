"""
Acquisition functions.
"""

from argparse import Namespace
import copy
import numpy as np
import tensorflow as tf
from copy import deepcopy
from collections import defaultdict
from scipy.stats import norm as sps_norm
from functools import partial
from tqdm import tqdm

from ..util.base import Base
from ..util.misc_util import dict_to_namespace, flatten
from ..util.timing import Timer
from ..models.function import FunctionSample
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
        self.params.name = getattr(params, 'name', 'AcqFunction')

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
        self.params.name = getattr(params, 'name', 'RandAcqFunction')

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
        self.params.name = getattr(params, 'name', 'AlgoAcqFunction')
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
        self.params.name = getattr(params, 'name', 'BaxAcqFunction')
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
        print('\t- clust_idx_list initial details:')
        len_list = [len(clust) for clust in cluster_idx_list]
        print(f'\t\t- min len_list: {np.min(len_list)},  max len_list: {np.max(len_list)},  len(len_list): {len(len_list)}')  # NOQA
        # -----

        # Filter clusters that are too small (looping to find optimal dist_thresh)
        min_nn = self.params.min_neighbors
        min_n_clust = self.params.min_n_clust
        min_dist_thresh = self.params.dist_thresh

        cluster_idx_list_new = [clust for clust in cluster_idx_list if len(clust) > min_nn]
        # -----
        print('\t- clust_idx_list_NEW details:')
        len_list = [len(clust) for clust in cluster_idx_list_new]
        print(f'\t\t- min len_list: {np.min(len_list)},  max len_list: {np.max(len_list)},  len(len_list): {len(len_list)}')
        # -----
        while len(cluster_idx_list_new) > min_n_clust and dist_thresh >= min_dist_thresh:
            cluster_idx_list_keep = cluster_idx_list_new
            dist_thresh -= self.params.dist_thresh_inc
            print(f'NOTE: dist_thresh = {dist_thresh}')
            cluster_idx_tmp = self.get_cluster_idx_list(output_list, dist_thresh)
            cluster_idx_list_new = [clust for clust in cluster_idx_tmp if len(clust) > min_nn]

        try:
            cluster_idx_list = cluster_idx_list_keep
        except UnboundLocalError:
            print(
                'WARNING: cluster_idx_list_keep not assigned, using cluster_idx_list.'
            )
            pass

        ## Only remove small clusters if there are enough big clusters
        #if len(cluster_idx_list_new) > self.params.min_n_clust:
            #cluster_idx_list = cluster_idx_list_new

        # -----
        len_list = [len(clust) for clust in cluster_idx_list]
        print('\t- clust_idx_list final details:')
        print(f'\t\t- min len_list: {np.min(len_list)},  max len_list: {np.max(len_list)},  len(len_list): {len(len_list)}')
        print(f'\t\tFound dist_thresh: {dist_thresh}')
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
            dist_sort = dist_sort[:self.params.max_neighbors]
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
        print('\t- clust_idx_list details:')
        len_list = [len(clust) for clust in cluster_idx_list]
        print(f'\t- min len_list: {np.min(len_list)},  max len_list: {np.max(len_list)},  len(len_list): {len(len_list)}')
        # -----

        ## Remove clusters that are too small
        #min_nn = self.params.min_neighbors
        #cluster_idx_list = [clust for clust in cluster_idx_list if len(clust) > min_nn]

        # -----
        len_list = [len(clust) for clust in cluster_idx_list]
        print(f'\t- min len_list: {np.min(len_list)},  max len_list: {np.max(len_list)},  len(len_list): {len(len_list)}')
        # -----

        # Compute entropies for posterior predictive given execution path samples
        h_samp_list = []
        #for samp_mean, samp_std in zip(samp_mean_list, samp_std_list):
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
            #weight_mat_den = np.ones(pdf_mat.shape)
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

            #weight_mat_num = np.ones(pdf_mat_sum.shape)
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

        with Timer(f"Compute acquisition function for a batch of {len(x_list)} points"):
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
            elif self.params.acq_str == 'out':
                acq_list = self.acq_out_normal(std, mu_list, std_list, self.output_list)
            elif self.params.acq_str == 'is':
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


class MesAcqFunction(BaxAcqFunction):
    """
    Class for max-value entropy search acquisition functions.
    """

    def set_params(self, params):
        """Set self.params, the parameters for the AcqFunction."""
        super().set_params(params)

        params = dict_to_namespace(params)
        self.params.name = getattr(params, 'name', 'MesAcqFunction')
        self.params.opt_mode = getattr(params, "opt_mode", "max")

    def get_acq_list_batch(self, x_list):
        """Return acquisition function for a batch of inputs x_list."""

        with Timer(f"Compute acquisition function for a batch of {len(x_list)} points"):
            # Compute entropies for posterior for x in x_list
            mu, std = self.model.get_post_mu_cov(x_list, full_cov=False)
            h_post = self.entropy_given_normal_std(std)

            mc_list = []
            for output in self.output_list:
                if self.params.opt_mode == "max":
                    gam = (output - np.array(mu)) / np.array(std)
                elif self.params.opt_mode == "min":
                    gam = (np.array(mu) - output) / np.array(std)
                t1 = gam * sps_norm.pdf(gam) / (2 * sps_norm.cdf(gam))
                t2 = np.log(sps_norm.cdf(gam))
                mc_list.append(t1 - t2)
            acq_list = np.mean(mc_list, 0)

        # Package and store acq_vars
        self.acq_vars = {
            "mu": mu,
            "std": std,
            "acq_list": acq_list,
        }

        # Return list of acquisition function on x in x_list
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
        self.params.name = getattr(params, 'name', 'RandBaxAcqFunction')

    def __call__(self, x_list):
        """Class is callable and returns acquisition function on x_list."""
        acq_list = super().__call__(x_list) # NOTE: would super()(x_list) work?
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
        self.params.name = getattr(params, 'name', 'UsBaxAcqFunction')

    def __call__(self, x_list):
        """Class is callable and returns acquisition function on x_list."""
        super().__call__(x_list) # NOTE: would super()(x_list) work?
        acq_list = self.acq_vars["std"]
        return acq_list


class EigfBaxAcqFunction(BaxAcqFunction):
    """
    Wrapper on BaxAcqFunction for EIG-on-f acquisition function, when we still want
    various BaxAcqFunction variables for visualizations.
    """

    def set_params(self, params):
        """Set self.params, the parameters for the AcqFunction."""
        super().set_params(params)

        params = dict_to_namespace(params)
        self.params.name = getattr(params, 'name', 'EigfBaxAcqFunction')

    def __call__(self, x_list):
        """Class is callable and returns acquisition function on x_list."""
        super().__call__(x_list) # NOTE: would super()(x_list) work?
        std_list = self.acq_vars["std"]
        acq_list = self.entropy_given_normal_std(std_list)
        return acq_list


class MultiBaxAcqFunction(AlgoAcqFunction):
    """
    Class for computing BAX acquisition functions.
    """

    def set_params(self, params):
        """Set self.params, the parameters for the AcqFunction."""
        super().set_params(params)

        params = dict_to_namespace(params)
        self.params.name = getattr(params, 'name', 'MultiBaxAcqFunction')

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
        with Timer(f"Compute acquisition function for a batch of {len(x_list)} points"):
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


class MultiSetBaxAcqFunction(AlgoAcqFunction):
    """
    Class for computing BAX acquisition functions.
    """

    def set_params(self, params):
        """Set self.params, the parameters for the AcqFunction."""
        super().set_params(params)

        params = dict_to_namespace(params)
        self.params.name = getattr(params, 'name', 'MultiBaxAcqFunction')
        self.smats = defaultdict(lambda: None)
        self.lmats = defaultdict(lambda: None)

    def set_model(self, model):
        """Set self.model, the model underlying the acquisition function."""
        if not model:
            raise ValueError("The model input parameter cannot be None.")
        else:
            self.model = copy.deepcopy(model)
            self.conditioning_model = copy.deepcopy(model)

    def acq_exe_normal(self, post_stds, samp_stds_list):
        """
        Since everything about this acquisition function is the same except for the log det of the covariance, it is simply much easier to do that
        """

        # Compute entropies for posterior predictive
        h_post = np.sum(np.linalg.slogdet(post_stds)[1])

        # Compute entropies for posterior predictive given execution path samples
        h_samp_list = []
        for samp_stds in samp_stds_list:
            dets = np.linalg.slogdet(samp_stds)[1]
            h_samp = np.sum(dets)
            h_samp_list.append(h_samp)

        avg_h_samp = np.mean(h_samp_list)
        acq_exe = h_post - avg_h_samp
        return acq_exe

    def get_acq_list_batch(self, x_list):
        """Return acquisition function for a batch of inputs x_set_list."""

        # Compute posterior, and post given each execution path sample, for x_list
        with Timer(f"Compute acquisition function for a batch of {len(x_list)} points"):
            acq_list = []
            # NOTE: self.model is multimodel so the following returns a list of mus and
            # a list of stds
            # going to implement this with a loop first, maybe we can make it more efficient later
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
                    self.conditioning_model.set_data(comb_data, lmats=self.lmats[i], smats=self.smats[i])
                    self.lmats[i] = self.conditioning_model.lmats
                    self.smats[i] = self.conditioning_model.smats

                    # NOTE: self.model is multimodel so the following returns a list of mus
                    # and a list of stds
                    samp_mus, samp_stds = self.model.get_post_mu_cov(
                        x_set, full_cov=True,
                    )
                    mus_list.append(samp_mus)
                    stds_list.append(samp_stds)

                # Compute acq_list, the acqfunction value for each x in x_list
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
        return acq_list

    def __call__(self, x_set_list):
        """Class is callable and returns acquisition function on x_set_list."""
        acq_list = self.get_acq_list_batch(x_set_list)
        return acq_list

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

        self.params.name = 'MCAcqFunction'
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
        self.params.name = getattr(params, 'name', 'UncertaintySamplingAcqFunction')

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
        with Timer(f"Compute acquisition function for a batch of {len(x_list)} points"):
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

class PILCOAcqFunction(AcqFunction):
    """
    Implements the KG for RL idea given in the overleaf.

    This version is a standard acquisition function that is only aiming to acquire
    a single state-action pair that maximizes the expected H-entropy gain
    for the cost function. To do this, it should run gradient descent on both the policy
    parameters and the point being acquired.

    This function is intended to be used for gradient-based acquisition and therefore
    doesn't compute the first term of the h-entropy.
    """
    def set_params(self, params):
        super().set_params(params)

        params = dict_to_namespace(params)
        self.params.num_fs = getattr(params, 'num_fs', 15)
        self.params.num_s0 = getattr(params, 'num_s0', 5)
        self.params.rollout_horizon = params.rollout_horizon
        self.params.p0 = params.p0
        self.params.reward_fn = params.reward_fn
        self.params.update_fn = params.update_fn
        self.params.gp_model_class = params.gp_model_class
        self.params.gp_model_params = params.gp_model_params
        self.verbose = getattr(params, "verbose", True)
        self.start_states = []

    def initialize(self):
        self.start_states = [self.params.p0() for _ in range(self.params.num_s0)]
        self.conditioned_model = self.params.gp_model_class(self.params.gp_model_params, self.model.data)
        self.rollout = partial(self.execute_policy_on_fs,
            model=self.conditioned_model,
            start_states=self.start_states,
            num_fs=self.params.num_fs,
            rollout_horizon=self.params.rollout_horizon,
            update_fn=self.params.update_fn,
            reward_fn=self.params.reward_fn)

    def __call__(self, policy_list, *args, **kwargs):
        '''
        Ignore the extra args so that we don't have to worry about anything besides the policies
        '''
        risks = []
        for policy in flatten(policy_list):
            neg_bayes_risk = self.rollout(policy, self.model.data.x, self.model.data.y)
            risks.append(neg_bayes_risk)
        return tf.reduce_mean(risks)

    @staticmethod
    def flatten(policy_list):
        out = []
        for item in policy_list:
            if type(item) is list:
                out += item
            else:
                out.append(item)
        return out

    @staticmethod
    def execute_policy_on_fs(
            policy,
            x_data,
            y_data,
            model,
            start_states,
            num_fs,
            rollout_horizon,
            update_fn,
            reward_fn,
            ):
        current_states = np.array(start_states)
        obs_dim = current_states.shape[-1]
        num_s0 = current_states.shape[0]
        data = Namespace(x=x_data, y=y_data)
        # don't need to sample from posterior predictive,
        # only function samples
        model.set_data(data, fs_only=True)
        model.initialize_function_sample_list(num_fs)
        current_states = np.repeat(current_states[np.newaxis, :, :], num_fs, axis=0)
        current_states = tf.convert_to_tensor(current_states, dtype=x_data.dtype)
        f_batch_list = model.call_function_sample_list
        returns = 0
        for t in range(rollout_horizon):
            current_states = tf.reshape(current_states, (-1, obs_dim))
            actions = policy(current_states)
            flat_x = tf.concat([current_states, actions], -1)
            x = tf.reshape(flat_x, (num_fs, num_s0, -1))
            deltas = f_batch_list(x)
            deltas = tf.reshape(deltas, (-1, obs_dim))
            current_states = update_fn(current_states, deltas)
            rewards = reward_fn(flat_x, current_states)
            rewards = tf.reshape(rewards, (num_fs, -1))
            returns = rewards + returns
        avg_return = tf.reduce_mean(returns)

        return avg_return

class KGRLAcqFunction(PILCOAcqFunction):
    """
    Implements the KG for RL idea given in the overleaf.

    This version is a standard acquisition function that is only aiming to acquire
    a single state-action pair that maximizes the expected H-entropy gain
    for the cost function. To do this, it should run gradient descent on both the policy
    parameters and the point being acquired.

    This function is intended to be used for gradient-based acquisition and therefore
    doesn't compute the first term of the h-entropy.
    """
    def set_params(self, params):
        super().set_params(params)
        params = dict_to_namespace(params)
        self.params.num_sprime_samps = getattr(params, 'num_sprime_samps', 5)

    def initialize(self):
        super().initialize()
        self.tf_acqfn = tf.function(partial(self.kgrl_acq,
            model=self.conditioned_model,
            num_sprime_samps=self.params.num_sprime_samps,
            start_states=self.start_states,
            num_fs=self.params.num_fs,
            rollout_horizon=self.params.rollout_horizon,
            update_fn=self.params.update_fn,
            reward_fn=self.params.reward_fn))

    def __call__(self, policy_list, x_list, lambdas):
        return self.tf_acqfn(policy_list, x_list, lambdas)

    @staticmethod
    def kgrl_acq(policy_list,
                 x_list,
                 lambdas,
                 model,
                 num_sprime_samps,
                 start_states,
                 num_fs,
                 rollout_horizon,
                 update_fn,
                 reward_fn):
        post_samples = model.sample_post_list(x_list, num_sprime_samps, lambdas)
        risks = []
        for i in range(post_samples.shape[0]):
            samp_batch = post_samples[i, ...]
            point_policies = policy_list[i]
            risk_samps = []
            for j in range(samp_batch.shape[0]):
                sprime = samp_batch[j, ...]
                new_x = x_list[i]
                policy = point_policies[j]
                x_data = tf.concat([model.data.x, new_x[None, :]], axis=0)
                y_data = tf.concat([model.data.y, sprime[None, :]], axis=0)
                neg_bayes_risk = KGRLAcqFunction.execute_policy_on_fs(policy,
                                                                      x_data,
                                                                      y_data,
                                                                      model,
                                                                      start_states,
                                                                      num_fs,
                                                                      rollout_horizon,
                                                                      update_fn,
                                                                      reward_fn)
                risk_samps.append(neg_bayes_risk)
            risks.append(tf.reduce_mean(risk_samps))

        return tf.reduce_mean(risks)


class KGRLPolicyAcqFunction(PILCOAcqFunction):
    """
    Implements the KG for RL idea given in the overleaf.

    This version is a standard acquisition function that is only aiming to acquire
    a single state-action pair that maximizes the expected H-entropy gain
    for the cost function. To do this, it should run gradient descent on both the policy
    parameters and the point being acquired.

    This function is intended to be used for gradient-based acquisition and therefore
    doesn't compute the first term of the h-entropy.
    """
    def set_params(self, params):
        super().set_params(params)
        params = dict_to_namespace(params)
        self.params.num_sprime_samps = getattr(params, 'num_sprime_samps', 5)
        self.params.planning_horizon = getattr(params, 'planning_horizon', 5)

    def initialize(self):
        super().initialize()
        self.tf_acqfn = tf.function(partial(self.kgrl_policy_acq,
            model=self.conditioned_model,
            num_sprime_samps=self.params.num_sprime_samps,
            start_states=self.start_states,
            num_fs=self.params.num_fs,
            rollout_horizon=self.params.rollout_horizon,
            update_fn=self.params.update_fn,
            reward_fn=self.params.reward_fn))

    def __call__(self, current_obs, policy_list, action_sequence, lambdas):
        return self.tf_acqfn(current_obs, policy_list, action_sequence, lambdas)

    @staticmethod
    def execute_actions_on_fs(action_sequence, model, current_obs, update_fn, num_fs):
        current_states = tf.repeat(current_obs[None, :], num_fs, axis=0)
        x_list = []
        y_list = []
        f_batch_list = model.call_function_sample_list
        for t in range(action_sequence.shape[0]):
            actions = tf.repeat(action_sequence[t:t+1, :], num_fs, axis=0)
            flat_x = tf.concat([current_states, actions], -1)
            deltas = tf.squeeze(f_batch_list(flat_x))
            x_list.append(flat_x)
            y_list.append(deltas)
            current_states = update_fn(current_states, deltas)
        x_batch = tf.stack(x_list)
        y_batch = tf.stack(y_list)
        return x_batch, y_batch

    @staticmethod
    def kgrl_policy_acq(current_obs,
                        policy_list,
                        action_sequence,
                        lambdas,
                        model,
                        num_sprime_samps,
                        start_states,
                        num_fs,
                        rollout_horizon,
                        update_fn,
                        reward_fn):
        model.initialize_function_sample_list(num_sprime_samps, weights=lambdas)
        # this needs to return num_sprime_samps x action_sequence.shape[0] x obs_dim tensor of all the xs and ys
        # observed in the execution
        sampled_new_data_x, sampled_new_data_y = KGRLPolicyAcqFunction.execute_actions_on_fs(action_sequence,
                                                                                             model,
                                                                                             current_obs,
                                                                                             update_fn,
                                                                                             num_sprime_samps)
        risks = []
        for i in range(num_sprime_samps):
            samp_batch_x = sampled_new_data_x[i, ...]
            samp_batch_y = sampled_new_data_y[i, ...]
            policy = policy_list[i]
            x_data = tf.concat([model.data.x, samp_batch_x], axis=0)
            y_data = tf.concat([model.data.y, samp_batch_y], axis=0)
            neg_bayes_risk = KGRLPolicyAcqFunction.execute_policy_on_fs(policy,
                                                                  x_data,
                                                                  y_data,
                                                                  model,
                                                                  start_states,
                                                                  num_fs,
                                                                  rollout_horizon,
                                                                  update_fn,
                                                                  reward_fn)
            risks.append(neg_bayes_risk)
        return tf.reduce_mean(risks)
