import numpy as np
import tensorflow as tf
import sys

from typing import Any
from gpflow.kernels.base import Product
from gpflow.kernels.stationaries import SquaredExponential
from gpflow.kernels.periodic import Periodic
from gpflow.base import TensorType
from gpflow.utilities import Dispatcher
from gpflow_sampling.bases.fourier_bases import AbstractFourierBasis
from gpflow.inducing_variables import InducingVariables
from gpflow_sampling.utils import (
    move_axis,
    expand_to,
    batch_tensordot,
    inducing_to_tensor,
)
from gpflow_sampling.bases import fourier_bases
from scipy.special import iv


# ---- Exports
__all__ = (
    "PeriodicBasis",
    "ProductBasis",
)

fourier_basis = Dispatcher("fourier_basis")


class PeriodicBasis(AbstractFourierBasis):
    def __init__(
        self,
        kernel: Periodic,
        num_bases: int,
        weights: tf.Tensor = None,
        biases: tf.Tensor = None,
        name: str = None,
    ):
        super().__init__(name=name, kernel=kernel, num_bases=num_bases)
        self._weights = weights
        self._biases = biases
        if not isinstance(kernel.base_kernel, SquaredExponential):
            raise ValueError(
                "Only squared exponential periodic kernels are supported for now."
            )

    def __call__(self, x: TensorType, **kwargs) -> tf.Tensor:
        if x.shape[-1] != 1:
            raise ValueError(
                "Only periodic kernels with one dimensional inputs are supported for now"
            )

        self._maybe_initialize(x, **kwargs)
        if isinstance(x, InducingVariables):  # TODO: Allow this behavior?
            x = inducing_to_tensor(x)

        proj = tf.tensordot(x, self.weights, axes=[-1, -1])  # [..., B]
        feat = tf.cos(proj + self.biases)
        return self.output_scale * feat

    def initialize(self, x: TensorType, dtype: Any = None):
        if isinstance(x, InducingVariables):
            x = inducing_to_tensor(x)

        if dtype is None:
            dtype = x.dtype

        self._biases = tf.random.uniform(
            shape=[self.num_bases], maxval=2 * np.pi, dtype=dtype
        )
        self._weights = self._init_weights()
        self._weights = tf.cast(self._weights, dtype)

    def _init_weights(self):
        p = []
        l = self.kernel.base_kernel.lengthscales

        n = 0
        csum = 0.0
        while True:
            pn = np.exp(-0.25 / l**2) * iv(n, 0.25 / l**2)
            if n > 0:
                pn *= 2
            csum += pn
            if pn / csum < 1e-4:
                break
            p.append(pn)
            n += 1

        p = np.array(p)
        p /= np.sum(p)

        omega = 2 * np.pi * tf.math.reciprocal(self.kernel.period)
        w = np.arange(n) * omega
        weights = np.random.choice(w, size=self.num_bases, replace=True, p=p)
        weights = weights.reshape((self.num_bases, 1))
        return weights

    @property
    def weights(self):
        return self._weights

    @property
    def biases(self):
        return self._biases

    @property
    def output_scale(self):
        return tf.sqrt(2 * self.kernel.base_kernel.variance / self.num_bases)


class ProductBasis(AbstractFourierBasis):
    def __init__(
        self,
        kernel: Product,
        num_bases: int,
        weights: tf.Tensor = None,
        biases: tf.Tensor = None,
        name: str = None,
    ):
        super().__init__(name=name, kernel=kernel, num_bases=num_bases)
        self._weights = weights
        self._biases = biases

    def __call__(self, x: TensorType, **kwargs) -> tf.Tensor:
        self._maybe_initialize(x, **kwargs)
        if isinstance(x, InducingVariables):  # TODO: Allow this behavior?
            x = inducing_to_tensor(x)

        x = tf.gather(x, self.perm_dim, axis=-1)
        proj = tf.tensordot(x, self.weights, axes=[-1, -1])  # [..., B]
        feat = tf.cos(proj + self.biases)
        return self.output_scale * feat

    def initialize(self, x: TensorType, dtype: Any = None):
        if isinstance(x, InducingVariables):
            x = inducing_to_tensor(x)

        if dtype is None:
            dtype = x.dtype

        weights = []
        self.perm_dim = []
        for k in self.kernel.kernels:
            x_sub = tf.gather(x, k.active_dims, axis=-1)
            basis = fourier_basis(k, num_bases=self.num_bases)
            basis(x_sub)
            weights.append(basis.weights)
            self.perm_dim.append(k.active_dims)

        self.perm_dim = tf.concat(self.perm_dim, axis=0)
        self._biases = tf.random.uniform(
            shape=[self.num_bases], maxval=2 * np.pi, dtype=dtype
        )
        self._weights = tf.concat(weights, axis=1)

        self.variance = 1.0

        for k in self.kernel.kernels:
            if isinstance(k, Periodic):
                variance = k.base_kernel.variance
            else:
                variance = k.variance
            self.variance = self.variance * variance

    @property
    def weights(self):
        return self._weights

    @property
    def biases(self):
        return self._biases

    @property
    def output_scale(self):
        return tf.sqrt(2 * self.variance / self.num_bases)
