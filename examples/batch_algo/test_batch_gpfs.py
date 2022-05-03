from argparse import Namespace
import numpy as np
from gpflow import kernels
from gpflow.config import default_float as floatx
import tensorflow as tf
from bax.models.gpfs.models import PathwiseGPR

# Set random seed
seed = 12
np.random.seed(seed)
tf.random.set_seed(seed)

# Set data
n_dimx = 2
data_x = [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]
data_y = [0.0, 1.0, 2.0]

tf_data_x = tf.convert_to_tensor(np.array(data_x))
tf_data_y = tf.convert_to_tensor(np.array(data_y).reshape(-1, 1))

# Set GP hypers and kernel
ls = 8.0
kernvar = 5.0**2
noisevar = 1e-2**2
gpf_kernel = kernels.SquaredExponential(variance=kernvar, lengthscales=ls)

# Set GPFS model
model = PathwiseGPR(
    data=(tf_data_x, tf_data_y), kernel=gpf_kernel, noise_variance=noisevar
)

# Set paths
n_fsamp = 3
n_batch = 2
n_bases = 100
paths = model.generate_paths(num_samples=n_fsamp, num_bases=n_bases)
_ = model.set_paths(paths)

Xinit = tf.random.uniform(
    [n_fsamp, n_batch, n_dimx], minval=0.0, maxval=0.1, dtype=floatx()
)
fsl_xvars = tf.Variable(Xinit)

# Make prediction
@tf.function
def call_model_predict_on_xvars(pred_model, xvars):
    """Call fsl on xvars."""
    fvals = pred_model.predict_f_samples(Xnew=xvars, sample_axis=0)
    return fvals


# x_list of inputs pts: [[sample_1: pt_1, pt_2, ...], [sample_2: pt_1, pt_2, ...], ...]
x_list = [
    [[0.1, 0.1], [0.1, 0.1]],
    [[10.0, 10.0], [9.8, 9.8]],
    [[10.0, 10.0], [9.8, 9.8]],
]

fsl_xvars.assign(x_list)

y_tf = call_model_predict_on_xvars(model, fsl_xvars)

y_npy = y_tf.numpy()
print("Final y_npy:")
print(y_npy)

print("Final list(y_npy):")
print([list(y.reshape(-1)) for y in y_npy])
