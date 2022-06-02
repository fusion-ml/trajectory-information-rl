from argparse import Namespace
import numpy as np
import matplotlib.pyplot as plt
from gpflow import kernels
from gpflow.config import default_float as floatx
import tensorflow as tf
from gpflow_sampling.models import PathwiseGPR
from bax.models.gpfs.periodic import Periodic


# Set random seed
seed = 12
np.random.seed(seed)
tf.random.set_seed(seed)

# Set data
n_dimx = 2
data_x = [[0.0], [-2.0], [7.0]]
data_y = [0.0, 1.0, 2.0]

tf_data_x = tf.convert_to_tensor(np.array(data_x))
tf_data_y = tf.convert_to_tensor(np.array(data_y).reshape(-1, 1))

# Set GP hypers and kernel
ls = 0.3
kernvar = 1.0**2
noisevar = 0.1**2
period = 4.0

kexp = kernels.SquaredExponential(variance=kernvar, lengthscales=ls, active_dims=[0])
kper = kernels.Periodic(kexp, period=period)
gpf_kernel = kernels.Product([kper])

# Set GPFS model
model = PathwiseGPR(
    data=(tf_data_x, tf_data_y), kernel=gpf_kernel, noise_variance=noisevar
)

# Set paths
n_fsamp = 3
n_batch = 3
n_bases = 20
paths = model.generate_paths(num_samples=n_fsamp, num_bases=n_bases)
_ = model.set_paths(paths)

xvars = np.linspace(-5, 10, 100).reshape((1, -1, 1)).repeat(n_fsamp, axis=0)
y_out = model.predict_f_samples(Xnew=xvars, sample_axis=0)
y_out = y_out.numpy()

for x, y in zip(xvars, y_out):
    plt.plot(x.flatten(), y.flatten(), c="b")
plt.scatter(data_x, data_y, c="r", s=100)
plt.show()
