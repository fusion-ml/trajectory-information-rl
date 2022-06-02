from argparse import Namespace

import numpy as np
import tensorflow as tf

from bax.models.gpflow_gp import GpflowGp, get_gpflow_hypers_from_data
from bax.util.domain_util import unif_random_sample_domain

from branin import branin, branin_xy


# Set seed
seed = 11
np.random.seed(seed)
tf.random.set_seed(seed)

# Set numpy print options
np.set_printoptions(suppress=True, precision=4)

# Set function
func = branin
n_dimx = 2

# Set domain
domain = [[-5, 10], [0, 15]]

# Set data for model
n_init_data = 10
data = Namespace()
data.x = unif_random_sample_domain(domain, n=n_init_data)
data.y = [func(x) for x in data.x]


def join_print_columns(col_list):
    arr_list = [np.array(col).reshape(-1, 1) for col in col_list]
    all_arr = np.concatenate(arr_list, axis=1)
    print(all_arr)


# Fit hyperparameters
gp_params = get_gpflow_hypers_from_data(data, print_fit_hypers=True)
print(f"gp_params = {gp_params}")
print("\n--------------------\n\n")

# Define two GpflowGp models
gpfp_params = dict(fixed_noise=False, sigma=0.1, print_fit_hypers=True)
gpfp = GpflowGp(gpfp_params, data, verbose=True)
gpfp_fixedsig_params = dict(fixed_noise=True, print_fit_hypers=True)
gpfp_fixedsig = GpflowGp(gpfp_fixedsig_params, data, verbose=True)

# Fit both models
gpfp.fit_hypers()
gpfp_fixedsig.fit_hypers()

# Test predictions on gpflow models
n_test_data = 50
x_test = unif_random_sample_domain(domain, n=n_test_data)
y_test = func(x_test)

y_pred_mean_fixedsig, y_pred_var_fixedsig = gpfp_fixedsig.get_post_mu_cov(
    x_test, full_cov=False
)
y_pred_mean, y_pred_var = gpfp.get_post_mu_cov(x_test, full_cov=False)

col_names = ("y_test", "y_pred_fixedsig", "y_pred", "2*std_fixedsig", "2*std")
print(f"Columns: {col_names}")
join_print_columns(
    [
        y_test,
        y_pred_mean_fixedsig,
        y_pred_mean,
        2 * np.sqrt(y_pred_var_fixedsig),
        2 * np.sqrt(y_pred_var),
    ]
)

# error_model = np.abs(y_test.reshape(-1) -  y_pred_mean.numpy().reshape(-1))
# error_model_fixedsig = np.abs(y_test.reshape(-1) -  y_pred_mean_fixedsig.numpy().reshape(-1))
error_gpfp = np.abs(y_test.reshape(-1) - y_pred_mean)
error_gpfp_fixedsig = np.abs(y_test.reshape(-1) - y_pred_mean_fixedsig)

print(f"mean(error_gpfp) = {np.mean(error_gpfp)}")
print(f"mean(error_gpfp_fixedsig) = {np.mean(error_gpfp_fixedsig)}")


# breakpoint()
