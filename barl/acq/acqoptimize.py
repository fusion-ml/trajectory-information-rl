"""
Code for optimizing acquisition functions.
"""
import copy
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tqdm import trange

from .acquisition import BaxAcqFunction
from ..util.base import Base
from ..util.misc_util import dict_to_namespace
from ..util.control_util import compute_return
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
        acq_list = self.acqfunction(x_batch)
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
                except:
                    break

        return x_batch

    def set_print_params(self):
        """Set self.print_params."""
        self.print_params = copy.deepcopy(self.params)


class KGAcqOptimizer(AcqOptimizer):
    def set_params(self, params):
        super().set_params(params)
        params = dict_to_namespace(params)
        self.params.learning_rate = getattr(params, 'learning_rate', 3e-4)
        self.params.num_steps = getattr(params, 'num_steps', 10000)
        self.params.obs_dim = params.obs_dim
        self.params.action_dim = params.action_dim
        self.params.hidden_layer_sizes = getattr(params, 'hidden_layer_sizes', [128, 128])
        self.params.num_sprime_samps = params.num_sprime_samps
        self.params.policy_test_period = params.policy_test_period
        self.params.num_eval_trials = params.num_eval_trials
        self.params.eval_fn = params.eval_fn
        self.params.tf_dtype = params.tf_dtype
        self.params.policies = getattr(params, 'policies', None)
        self.risk_vals = None
        self.eval_vals = None
        self.eval_steps = None
        self.tf_train_step = tf.function(self.train_step)

    @staticmethod
    def get_policies(num_x, num_sprime_samps, obs_dim, action_dim, hidden_layer_sizes):
        policies = []
        for _ in range(num_x):
            xval_policies = []
            for __ in range(num_sprime_samps):
                xval_policies.append(TanhMlpPolicy(obs_dim, action_dim, hidden_layer_sizes))
            policies.append(xval_policies)
        return policies

    @staticmethod
    def train_step(acqfn, opt, policies, x_batch, lambdas):
        opt_vars = [x_batch]
        for x_policies in policies:
            for policy in x_policies:
                opt_vars += policy.trainable_variables
        with tf.GradientTape() as tape:
            loss_val = -1 * acqfn(policies, x_batch, lambdas)
        grads = tape.gradient(loss_val, opt_vars)
        clipped_grads = [tf.clip_by_norm(grad, clip_norm=10) for grad in grads]
        opt.apply_gradients(zip(clipped_grads, opt_vars))
        x_batch.assign(tf.clip_by_value(x_batch, -1, 1))
        return loss_val

    def optimize(self, x_batch):
        x_batch = tf.Variable(x_batch, dtype=self.params.tf_dtype)
        lambdas = tf.random.normal((x_batch.shape[0], self.params.num_sprime_samps, self.params.obs_dim), dtype=self.params.tf_dtype)
        # policies = [[TanhMlpPolicy(self.params.obs_dim, self.params.action_dim, self.params.hidden_layer_sizes) for _ in range(x_batch.shape[0])]
        opt = keras.optimizers.Adam(learning_rate=self.params.learning_rate)
        if self.params.policies is None:
            self.params.policies = self.get_policies(x_batch.shape[0], self.params.num_sprime_samps, self.params.obs_dim, self.params.action_dim,
                                                     self.params.hidden_layer_sizes)
        self.risk_vals = []
        self.eval_vals = []
        self.eval_steps = []
        pbar = trange(self.params.num_steps)
        best_risk = np.inf
        avg_return = None
        optima = None
        for i in pbar:
            if self.params.policy_test_period != 0 and i % self.params.policy_test_period == 0:
                self.eval_steps.append(i)
                avg_return = self.evaluate(self.params.policies)
            bayes_risk = float(self.tf_train_step(self.acqfunction, opt, self.params.policies, x_batch, lambdas))
            if bayes_risk < best_risk:
                best_risk = bayes_risk
                optima = np.squeeze(x_batch.numpy())
            self.risk_vals.append(bayes_risk)
            postfix = {"Bayes Risk": bayes_risk}
            if avg_return is not None:
                postfix["Avg Return"] = avg_return
            pbar.set_postfix(postfix)
        return optima, best_risk

    def evaluate(self, policies):
        '''
        Evaluate the policy here on the real environment. This information should not
        be used by the optimizer, it is strictly for diagnostic purposes.
        '''
        all_returns = []
        policies = [policy for plist in policies for policy in plist]
        for policy in policies:
            policy_returns = []
            for i in range(self.params.num_eval_trials):
                obs, actions, rewards = self.params.eval_fn(policy)
                returns = compute_return(rewards, 1)
                policy_returns.append(returns)
            all_returns.append(policy_returns)
        self.eval_vals.append(all_returns)
        return np.mean(all_returns)


class KGPolicyAcqOptimizer(KGAcqOptimizer):
    def set_params(self, params):
        super().set_params(params)
        params = dict_to_namespace(params)
        self.params.num_bases = getattr(params, 'num_bases', 1000)
        self.params.planning_horizon = getattr(params, 'planning_horizon', 5)
        self.tf_train_step = tf.function(self.train_step)

    @staticmethod
    def get_policies(num_sprime_samps, obs_dim, action_dim, hidden_layer_sizes):
        policies = []
        for _ in range(num_sprime_samps):
            policies.append(TanhMlpPolicy(obs_dim, action_dim, hidden_layer_sizes))
        return policies

    def sample_action_sequence(self, horizon):
        # assumes action_sequence is in [-1, 1]^{planning_horizon x action_dim}
        action_sequence = tf.Variable(tf.random.uniform([horizon, self.params.action_dim],
                                             dtype=self.params.tf_dtype) * 2 - 1)
        return action_sequence

    def advance_action_sequence(self, action_sequence, current_obs):
        current_action = action_sequence[0, :]
        x = np.concatenate([current_obs, current_action], axis=-1)
        new_action = self.sample_action_sequence(1)
        # this might crash idk
        return x, tf.Variable(tf.concat([action_sequence[1:, :], new_action], axis=0))


    @staticmethod
    def train_step(acqfn, opt, current_obs, action_sequence, policies, lambdas):
        opt_vars = [action_sequence]
        for policy in policies:
            opt_vars += policy.trainable_variables
        with tf.GradientTape() as tape:
            loss_val = -1 * acqfn(current_obs, policies, action_sequence, lambdas)
        grads = tape.gradient(loss_val, opt_vars)
        clipped_grads = [tf.clip_by_norm(grad, clip_norm=10) for grad in grads]
        opt.apply_gradients(zip(clipped_grads, opt_vars))
        action_sequence.assign(tf.clip_by_value(action_sequence, -1, 1))
        return loss_val

    def optimize(self, x_batch):
        # assume x_batch is 1x(obs_dim + action_dim)
        current_obs = tf.Tensor(x_batch[0, :obs_dim], dtype=self.params.tf_dtype)
        lambdas = tf.random.normal((self.params.obs_dim, self.params.num_sprime_samps, 1, self.params.num_bases),
                                   dtype=self.params.tf_dtype)
        opt = keras.optimizers.Adam(learning_rate=self.params.learning_rate)
        if self.params.policies is None:
            self.params.policies = self.get_policies(self.params.num_sprime_samps, self.params.obs_dim,
                                                     self.params.action_dim, self.params.hidden_layer_sizes)
        if self.params.action_sequence is None:
            self.params.action_sequence = self.sample_action_sequence()
        self.risk_vals = []
        self.eval_vals = []
        self.eval_steps = []
        pbar = trange(self.params.num_steps)
        best_risk = np.inf
        avg_return = None
        optima = None
        for i in pbar:
            if self.params.policy_test_period != 0 and i % self.params.policy_test_period == 0:
                self.eval_steps.append(i)
                avg_return = self.evaluate(self.params.policies)
            bayes_risk = float(self.tf_train_step(self.acqfunction,
                                                  opt,
                                                  current_obs,
                                                  self.params.action_sequence,
                                                  self.params.policies,
                                                  x_batch,
                                                  lambdas))
            if bayes_risk < best_risk:
                best_risk = bayes_risk
                optimum = self.params.action_sequence.numpy()
            self.risk_vals.append(bayes_risk)
            postfix = {"Bayes Risk": bayes_risk}
            if avg_return is not None:
                postfix["Avg Return"] = avg_return
            pbar.set_postfix(postfix)
        optimum, self.params.action_sequence = self.advance_action_sequence(optimum, current_obs)
        return optimum, best_risk
