"""
Model-Predictive control using Bayes risk.
Searches over a variety of posterior samples from the dynamics
to find an action that minimizes reward.
"""

import logging
import numpy as np
from tqdm import trange


from ..util.base import Base
from ..util.control_util import iCEM_generate_samples
from ..util.misc_util import dict_to_namespace


class BayesMPCPolicy(Base):
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
        self.params.planning_horizon = max(params.planning_horizon, 2)
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
        self.params.update_fn = params.update_fn
        self.params.reward_fn = params.reward_fn
        self.params.function_sample_list = getattr(params, "function_sample_list", None)

    @property
    def function_sample_list(self):
        return self.params.function_sample_list

    @function_sample_list.setter
    def function_sample_list(self, fsl):
        self.params.function_sample_list = fsl

    def get_initial_mean(self, action_sequence):
        if action_sequence is None:
            return np.zeros((self.params.planning_horizon, self.params.action_dim))
        else:
            new_action_sequence = np.concatenate(
                [action_sequence[1:, :], np.zeros((1, self.params.action_dim))], axis=0
            )
            return new_action_sequence[: self.params.planning_horizon, ...]

    def __call__(self, current_obs, **kwargs):
        """
        kwargs are for other policies that need other state. They should be a noop here.
        """
        # assume x_batch is (obs_dim,)
        horizon = self.params.planning_horizon
        beta = self.params.beta
        mean = self.get_initial_mean(self.params.action_sequence)
        if self.params.actions_until_plan > 0:
            self.params.action_sequence = mean
            action = mean[0, :]
            self.params.actions_until_plan -= 1
            return mean

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

        action = best_sample[0, :]
        self.params.action_sequence = best_sample
        # subtract one since we are already doing this action
        self.params.actions_until_plan = self.params.actions_per_plan - 1
        return action

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
        f_batch_list = self.function_sample_list
        x_list = []
        sample_returns = np.zeros((num_samples,))
        for t in range(self.params.planning_horizon):
            actions = samples[:, :, t, :]
            # now actions is num_fs x num_samples x action_dim
            x = np.concatenate([current_obs, actions], axis=-1)
            x_list.append(x)
            deltas = f_batch_list(x)
            # might need to flatten tensors for this
            current_obs = self.params.update_fn(current_obs, deltas)
            # rewards is a matrix num_samples x num_fs
            rewards = self.params.reward_fn(x, current_obs)
            # average across function samples and add to sample_rewturns
            sample_returns += rewards.mean(axis=0)
        return sample_returns
