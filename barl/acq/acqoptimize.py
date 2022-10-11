"""
Code for optimizing acquisition functions.
"""
import copy
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import logging
from math import ceil
from tqdm import trange

from .acquisition import BaxAcqFunction
from ..util.timing import Timer
from ..util.base import Base
from ..util.misc_util import dict_to_namespace, flatten
from ..util.control_util import compute_return, iCEM_generate_samples
from ..policies import TanhMlpPolicy


class AcqOptimizer(Base):
    """
    Class for optimizing acquisition functions.
    """

    def set_params(self, params):
        """Set self.params, the parameters for the AcqOptimizer."""
        super().set_params(params)
        params = dict_to_namespace(params)

        self.params.name = getattr(params, "name", "AcqOptimizer")
        self.params.opt_str = getattr(params, "opt_str", "batch")
        # default_x_batch = [[x] for x in np.linspace(0.0, 40.0, 500)]
        # self.params.x_batch = getattr(params, "x_batch", default_x_batch)
        # self.params.x_batch = params.x_batch
        self.params.remove_x_dups = getattr(params, "remove_x_dups", False)
        self.params.max_batch_size = getattr(params, "max_batch_size", 200)

    def initialize(self, acqfunction):
        # Set self.acqfunction
        self.set_acqfunction(acqfunction)
        self.acqfunction.initialize()

    def optimize(self, x_batch):
        """
        Optimize acquisition function.

        Parameters
        ----------
        acqfunction : AcqFunction
            AcqFunction instance.
        """
        self.params.x_batch = x_batch

        with Timer(
            "Optimize acquisition function using random shooting", level=logging.INFO
        ):
            if self.params.opt_str == "batch":
                acq_opt, acq_val = self.optimize_batch()

            return acq_opt, acq_val

    def set_acqfunction(self, acqfunction):
        """Set self.acqfunction, the acquisition function."""
        if not acqfunction:
            # If acqfunction is None, set default acqfunction as BaxAcqFunction
            params = {"acq_str": "out"}
            self.acqfunction = BaxAcqFunction(params)
        else:
            self.acqfunction = acqfunction

    def optimize_batch(self):
        """Optimize acquisition function over self.params.x_batch."""
        x_batch = copy.deepcopy(self.params.x_batch)

        # Optionally remove data.x (in acqfunction) duplicates
        if self.params.remove_x_dups:
            x_batch = self.remove_x_dups(x_batch)

        # Optimize self.acqfunction over x_batch
        nbatches = ceil(len(x_batch) / self.params.max_batch_size)
        acq_list = []
        for i in trange(nbatches):
            minibatch = x_batch[
                (i * self.params.max_batch_size) : (
                    (i + 1) * self.params.max_batch_size
                )
            ]
            acq_list += list(self.acqfunction(minibatch))
        acq_idx = np.argmax(acq_list)
        acq_opt = x_batch[acq_idx]
        acq_val = acq_list[acq_idx]

        return acq_opt, acq_val

    def remove_x_dups(self, x_batch):
        """Remove elements of x_batch that are also in data.x (in self.acqfunction)"""

        # NOTE this requires self.acqfunction with model.data
        data = self.acqfunction.model.data

        # NOTE this only works for data.x consisting of list-types, not for arbitrary data.x
        for x in data.x:
            while True:
                try:
                    idx, pos = next(
                        (tup for tup in enumerate(x_batch) if all(tup[1] == x))
                    )
                    del x_batch[idx]
                except Exception:
                    break

        return x_batch

    def set_print_params(self):
        """Set self.print_params."""
        self.print_params = copy.deepcopy(self.params)


class PolicyAcqOptimizer(AcqOptimizer):
    """
    An optimizer that finds an action sequence that optimizes the JointSetBaxAcqFunction
    uses posterior function samples in order to figure out what samples will come from future actions.
    """

    def set_params(self, params):
        super().set_params(params)
        params = dict_to_namespace(params)
        self.params.obs_dim = params.obs_dim
        self.params.action_dim = params.action_dim
        self.params.initial_variance_divisor = getattr(
            params, "initial_variance_divisor", 4
        )
        self.params.base_nsamps = params.base_nsamps
        self.params.planning_horizon = max(
            min(params.planning_horizon, params.time_left), 2
        )
        self.params.n_elites = params.n_elites
        self.params.beta = params.beta
        self.params.gamma = params.gamma
        self.params.xi = params.xi
        self.params.num_fs = params.num_fs
        self.params.num_iters = params.num_iters
        self.params.verbose = getattr(params, "verbose", False)
        self.params.actions_per_plan = getattr(params, "actions_per_plan", 1)
        self.params.actions_until_plan = getattr(params, "actions_until_plan", 0)
        self.params.action_sequence = getattr(params, "action_sequence", None)
        self.params.action_upper_bound = getattr(params, "action_upper_bound", 1)
        self.params.action_lower_bound = getattr(params, "action_lower_bound", -1)
        self.params.num_s0_samps = params.num_s0_samps
        self.params.s0_sampler = params.s0_sampler
        self.params.update_fn = params.update_fn

    def initialize(self, acqfunction):
        # Set self.acqfunction
        self.set_acqfunction(acqfunction)
        if self.params.actions_until_plan == 0:
            self.acqfunction.initialize()
            self.acqfunction.model.initialize_function_sample_list(self.params.num_fs)

    def get_initial_mean(self, action_sequence):
        if action_sequence is None:
            return np.zeros((self.params.planning_horizon, self.params.action_dim))
        else:
            new_action_sequence = np.concatenate(
                [action_sequence[1:, :], np.zeros((1, self.params.action_dim))], axis=0
            )
            return new_action_sequence[: self.params.planning_horizon, ...]

    def optimize(self, x_batch=None):
        # wrapper so we can add timing info
        with Timer(
            "Optimize acquisition function using cross-entropy", level=logging.INFO
        ):
            if x_batch is not None:
                return self._optimize(x_batch)
            best_query, best_action_sequence, best_return = None, None, -np.inf
            for i in range(self.params.num_s0_samps):
                s0 = self.params.s0_sampler()
                optimum, value = self._optimize([s0])
                if value > best_return:
                    best_query = optimum
                    best_return = value
                    best_action_sequence = self.params.action_sequence
            self.params.action_sequence = best_action_sequence
            return best_query, best_return

    def _optimize(self, x_batch):
        # assume x_batch is 1x(obs_dim + action_dim)
        current_obs = x_batch[0][: self.params.obs_dim]
        horizon = self.params.planning_horizon
        beta = self.params.beta
        mean = self.get_initial_mean(self.params.action_sequence)
        if self.params.actions_until_plan > 0:
            self.params.action_sequence = mean
            query = np.concatenate([current_obs, mean[0, :]])
            self.params.actions_until_plan -= 1
            return query, 0.0

        initial_variance_divisor = 4
        action_upper_bound = self.params.action_upper_bound
        action_lower_bound = self.params.action_lower_bound
        var = (
            np.ones_like(mean)
            * ((action_upper_bound - action_lower_bound) / initial_variance_divisor)
            ** 2
        )

        elites, elite_returns = None, None
        best_sample, best_return = None, -np.inf
        for i in trange(self.params.num_iters, disable=not self.params.verbose):
            # these are num_samples x horizon x action_dim
            samples = iCEM_generate_samples(
                self.params.base_nsamps,
                horizon,
                beta,
                mean,
                var,
                action_lower_bound,
                action_upper_bound,
            )
            if i == 0:
                pass
                # do we need to do something specific here?
            if i + 1 == self.params.num_iters:
                samples = np.concatenate([samples, mean[None, :]], axis=0)
                samples = samples[1:, ...]
            returns = self.evaluate_samples(current_obs, samples)
            if i > 0:
                elite_subset_idx = np.random.choice(
                    self.params.n_elites,
                    int(self.params.n_elites * self.params.xi),
                    replace=False,
                )
                elite_subset = elites[elite_subset_idx, ...]
                elite_return_subset = elite_returns[elite_subset_idx]
                samples = np.concatenate([samples, elite_subset], axis=0)
                returns = np.concatenate([returns, elite_return_subset])
            elite_idx = np.argsort(returns)[-self.params.n_elites :]
            elites = samples[elite_idx, ...]
            elite_returns = returns[elite_idx]
            mean = np.mean(elites, axis=0)
            var = np.var(elites, axis=0)
            best_idx = np.argmax(returns)
            best_current_return = returns[best_idx]
            logging.debug(f"{best_current_return=}")
            if best_current_return > best_return:
                best_return = best_current_return
                best_sample = samples[best_idx, ...]

        optimum = np.concatenate([current_obs, best_sample[0, :]])
        self.params.action_sequence = best_sample
        # subtract one since we are already doing this action
        self.params.actions_until_plan = self.params.actions_per_plan - 1
        return optimum, best_return

    def evaluate_samples(self, current_obs, samples):
        # samples are initially num_samples x horizon x action_dim
        # num_fs x num_samples x obs_dim
        num_samples = samples.shape[0]
        horizon = samples.shape[1]
        assert horizon == self.params.planning_horizon
        current_obs = np.tile(current_obs, (self.params.num_fs, num_samples, 1))
        # current_obs = current_obs.reshape((-1, self.params.obs_dim))
        # num_fs x num_samples x horizon x action_dim
        samples = np.tile(samples, (self.params.num_fs, 1, 1, 1))
        f_batch_list = self.acqfunction.model.call_function_sample_list
        x_list = []
        for t in range(self.params.planning_horizon):
            actions = samples[:, :, t, :]
            # now actions is num_fs x num_samples x action_dim
            x = np.concatenate([current_obs, actions], axis=-1)
            x_list.append(x)
            deltas = f_batch_list(x)
            # might need to flatten tensors for this
            current_obs = self.params.update_fn(current_obs, deltas)
        # 4d tensor horizon x num_fs x num_samples x (obs_dim + action_dim)
        x_list = np.array(x_list)
        # num_samples x num_fs x horizon x (obs_dim + action_dim)
        x_list = x_list.transpose((2, 1, 0, 3))
        # (num_samples x num_fs) x horizon x (obs_dim + action_dim)
        x_list = x_list.reshape(
            (-1, horizon, self.params.obs_dim + self.params.action_dim)
        )
        acqvals = self.acqfunction(x_list)
        acqvals = np.mean(
            np.array(acqvals).reshape((num_samples, self.params.num_fs)), axis=1
        )
        return acqvals
