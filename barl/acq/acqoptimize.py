"""
Code for optimizing acquisition functions.
"""

import copy
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tqdm import trange, tqdm

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
        self.risk_vals = None
        self.eval_vals = None
        self.eval_steps = None

    def optimize(self, x_batch):
        x_batch = tf.Variable(x_batch, dtype=tf.float64)
        policies = []
        lambdas = tf.random.normal((x_batch.shape[0], self.params.num_sprime_samps, self.params.obs_dim), dtype=tf.float64)
        for _ in range(x_batch.shape[0]):
            xval_policies = []
            for __ in range(self.params.num_sprime_samps):
                xval_policies.append(TanhMlpPolicy(self.params.obs_dim, self.params.action_dim, self.params.hidden_layer_sizes))
            policies.append(xval_policies)
        # policies = [[TanhMlpPolicy(self.params.obs_dim, self.params.action_dim, self.params.hidden_layer_sizes) for _ in range(x_batch.shape[0])]
        opt = keras.optimizers.Adam(learning_rate=self.params.learning_rate)

        def loss():
            return -1 * self.acqfunction(policies, x_batch, lambdas)
        # opt_vars = [x_batch]
        self.risk_vals = []
        self.eval_vals = []
        self.eval_steps = []
        opt_vars = []
        for x_policies in policies:
            for policy in x_policies:
                opt_vars += policy.trainable_variables
        pbar = trange(self.params.num_steps)
        avg_return = None
        for i in pbar:
            if self.params.policy_test_period != 0 and i % self.params.policy_test_period == 0:
                self.eval_steps.append(i)
                avg_return = self.evaluate(policies)
            with tf.GradientTape() as tape:
                loss_val = loss()
            self.risk_vals.append(float(loss_val))
            grads = tape.gradient(loss_val, opt_vars)
            # tqdm.write(f"{tf.reduce_max(tf.abs(grads[0]))=}")
            opt.apply_gradients(zip(grads, opt_vars))
            # TODO: make sure we're in a NormalizedBoxEnv or use other bounds
            x_batch.assign(tf.clip_by_value(x_batch, -1, 1))
            # tqdm.write(f"{x_batch.numpy()=}")
            postfix = {"Bayes Risk": loss_val.numpy()}
            if avg_return is not None:
                postfix["Avg Return"] = avg_return
            pbar.set_postfix(postfix)
        optima = np.squeeze(x_batch.numpy())
        final_losses = loss()
        return optima, np.squeeze(final_losses.numpy())

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

    def optimize_batch(self):
        raise NotImplementedError()
