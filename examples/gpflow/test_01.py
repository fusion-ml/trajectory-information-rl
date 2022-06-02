from argparse import Namespace

import numpy as np
import gpflow
import tensorflow as tf

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
data = Namespace()
n_init_data = 10
data.x = unif_random_sample_domain(domain, n=n_init_data)
data.y = [func(x) for x in data.x]
gpf_data = (np.array(data.x).reshape(-1, n_dimx), np.array(data.y).reshape(-1, 1))

# Set mean function
mean_func = gpflow.mean_functions.Constant()
mean_func.c.assign([0.0])
gpflow.utilities.set_trainable(mean_func.c, False)

# Set kernels
hypers = Namespace(ls=[1.0, 1.0], var=1.0)
kernel = gpflow.kernels.SquaredExponential(variance=hypers.var, lengthscales=hypers.ls)
kernel_fixedsig = gpflow.kernels.SquaredExponential(
    variance=hypers.var, lengthscales=hypers.ls
)

# Set GPR models
model = gpflow.models.GPR(data=gpf_data, kernel=kernel, mean_function=mean_func)
model.likelihood.variance.assign(0.01)

model_fixedsig = gpflow.models.GPR(
    data=gpf_data, kernel=kernel_fixedsig, mean_function=mean_func
)
model_fixedsig.likelihood.variance.assign(0.1)
gpflow.utilities.set_trainable(model_fixedsig.likelihood.variance, False)

gpflow.utilities.print_summary(model)
gpflow.utilities.print_summary(model_fixedsig)


def join_print_columns(col_list):
    arr_list = [np.array(col).reshape(-1, 1) for col in col_list]
    all_arr = np.concatenate(arr_list, axis=1)
    print(all_arr)


opt = gpflow.optimizers.Scipy()
opt_config = dict(maxiter=1000)
print("Start training: model")
opt_log = opt.minimize(
    model.training_loss, model.trainable_variables, options=opt_config
)
print("End training: model")


opt = gpflow.optimizers.Scipy()
print("Start training: model_fixedsig")
opt_log = opt.minimize(
    model_fixedsig.training_loss, model_fixedsig.trainable_variables, options=opt_config
)
print("End training: model_fixedsig")

gpflow.utilities.print_summary(model)
gpflow.utilities.print_summary(model_fixedsig)


n_test_data = 50
x_test = unif_random_sample_domain(domain, n=n_test_data)
y_test = func(x_test)

x_test_gpf = np.array(x_test).reshape(-1, n_dimx)
y_pred_mean_fixedsig, y_pred_var_fixedsig = model_fixedsig.predict_f(x_test_gpf)
y_pred_mean, y_pred_var = model.predict_f(x_test_gpf)

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

error_model = np.abs(y_test.reshape(-1) - y_pred_mean.numpy().reshape(-1))
error_model_fixedsig = np.abs(
    y_test.reshape(-1) - y_pred_mean_fixedsig.numpy().reshape(-1)
)

print(f"mean(error_model) = {np.mean(error_model)}")
print(f"mean(error_model_fixedsig) = {np.mean(error_model_fixedsig)}")


# breakpoint()
