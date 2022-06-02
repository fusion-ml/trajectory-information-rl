from argparse import Namespace
import numpy as np
import matplotlib.pyplot as plt
from gpflow import kernels
import tensorflow as tf
from gpflow_sampling.models import PathwiseGPR
from bax.models.gpfs.periodic import Periodic


# Set random seed
seed = 12
np.random.seed(seed)
tf.random.set_seed(seed)

# Set data
data_x = [[0.0, 0.1, 0.0], [1.0, 0.0, 1.0], [2.0, 1.0, 2.0]]
data_y = [3.0, 1.0, 2.0]

tf_data_x = tf.convert_to_tensor(np.array(data_x))
tf_data_y = tf.convert_to_tensor(np.array(data_y).reshape(-1, 1))

# Set GP hypers and kernel
kernvar = 5.0**2
noisevar = 1e-2**2
period = 4.0

kexp1 = kernels.SquaredExponential(
    variance=kernvar, lengthscales=[1.0, 1.5], active_dims=[0, 1]
)
kexp2 = kernels.SquaredExponential(variance=1.0, lengthscales=1.0, active_dims=[2])
kper = kernels.Periodic(kexp2, period=period)
gpf_kernel = kernels.Product([kexp1, kper])

# Set GPFS model
model = PathwiseGPR(
    data=(tf_data_x, tf_data_y), kernel=gpf_kernel, noise_variance=noisevar
)

# Set paths
n_fsamp = 2
n_batch = 3
n_bases = 5
paths = model.generate_paths(num_samples=n_fsamp, num_bases=n_bases)
_ = model.set_paths(paths)

# x_list of inputs pts: [[sample_1: pt_1, pt_2, ...], [sample_2: pt_1, pt_2, ...], ...]
x_list = [
    [[0.0, 0.0, 0.0], [0.0, 1.0, 4.0], [0.01, 0.0, 8.0]]
]  # , [[10.0, 10.0], [10.0, 6.0]], [[10.0, 10.0], [9.8, 9.8]]]
xvars = np.array(x_list)

y_tf = model.predict_f_samples(Xnew=xvars, sample_axis=0)

y_npy = y_tf.numpy()
print("Final y_npy:")
print(y_npy)

print("Final list(y_npy):")
print([list(y.reshape(-1)) for y in y_npy])
