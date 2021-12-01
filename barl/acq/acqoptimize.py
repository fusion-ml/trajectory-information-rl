"""
Code for optimizing acquisition functions.
"""

import copy
import numpy as np
import tensorflow as tf
import tf.keras as keras

from .acquisition import BaxAcqFunction
from ..util.base import Base
from ..util.misc_util import dict_to_namespace
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


class GDAcqOptimizer(AcqOptimizer):
    def set_params(self, params):
        super().set_params(params)
        self.params.learning_rate = getattr(params, 'learning_rate', 3e-4)
        self.params.num_steps = getattr(params, 'num_steps', 10000)
        self.params.obs_dim = params.obs_dim
        self.params.action_dim = params.action_dim
        self.params.hidden_layer_sizes = getattr(params, 'hidden_layer_sizes', [128, 128])

    def optimize(self, x_batch):
        x_batch = tf.Variable(x_batch)
        policies = [TanhMlpPolicy(self.params.obs_dim, self.params.action_dim, self.params.hidden_layer_sizes) for _ in range(x_batch.shape[0])]
        opt = keras.optimizers.Adam(learning_rate=self.params.learning_rate)

        def loss():
            return -1 * self.acqfunction(policies, x_batch)
        opt_vars = [x_batch] + [policy.model.trainable_variables for policy in policies]
        for _ in range(self.params.num_steps):
            opt.minimize(loss, opt_vars)
        optima = x_batch.numpy()
        final_losses = loss()
        return optima

    def optimize_batch(self):
        raise NotImplementedError()
