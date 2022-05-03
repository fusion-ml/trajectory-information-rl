"""
Testing MultiGpfsGp and MultiBaxAcqFunction classes
"""

from argparse import Namespace
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from tqdm import trange
import tensorflow as tf

from bax.models.gpfs_gp import MultiGpfsGp
from bax.acq.acquisition import MultiBaxAcqFunction
from bax.acq.acqoptimize import AcqOptimizer
from bax.alg.mpc import MPC
from bax.util.misc_util import dict_to_namespace
from bax.util.envs.goddard import GoddardEnv, goddard_reward, goddard_terminal
from bax.util.control_util import ResettableEnv, get_f_mpc, compute_return
from bax.util.domain_util import unif_random_sample_domain, project_to_domain
import neatplot


# Set plot settings
neatplot.set_style()
neatplot.update_rc("figure.dpi", 120)
neatplot.update_rc("text.usetex", False)


# Set random seed
seed = 11
np.random.seed(seed)
tf.random.set_seed(seed)


def plot_path_2d(path, ax=None, true_path=False):
    """Plot a path through an assumed two-dimensional state space."""
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))

    x_plot = [xi[0] for xi in path.x]
    y_plot = [xi[1] for xi in path.x]

    if true_path:
        ax.plot(x_plot, y_plot, "k--", linewidth=3)
        ax.plot(x_plot, y_plot, "*", color="k", markersize=5)
    else:
        ax.plot(x_plot, y_plot, "k--", linewidth=1, alpha=0.3)
        ax.plot(x_plot, y_plot, "o", alpha=0.3)


# -------------
# Start Script
# -------------
# Set black-box function
env = GoddardEnv()
env.seed(seed)
obs_dim = env.observation_space.low.size
action_dim = env.action_space.low.size
plan_env = GoddardEnv()
plan_env.seed(seed)
f = get_f_mpc(plan_env)
start_obs = env.reset()

# Set domain
low = np.concatenate([env.observation_space.low, env.action_space.low])
high = np.concatenate([env.observation_space.high, env.action_space.high])
domain = [elt for elt in zip(low, high)]

# Set algorithm
algo_class = MPC
algo_params = dict(
    start_obs=start_obs,
    env=plan_env,
    reward_function=goddard_reward,
    terminal_function=goddard_terminal,
    base_nsamps=20,
    planning_horizon=30,
    n_elites=6,
    beta=3,
    gamma=1.25,
    xi=0.3,
    num_iters=3,
    actions_per_plan=4,
    project_to_domain=True,
    domain=domain,
)
algo = algo_class(algo_params)

# Set model
gp_params = {"ls": 2.0, "alpha": 2.0, "sigma": 1e-2, "n_dimx": obs_dim + action_dim}
multi_gp_params = {"n_dimy": obs_dim, "gp_params": gp_params}
gp_model_class = MultiGpfsGp

# Set data
data = Namespace()
n_init_data = 5000
data.x = unif_random_sample_domain(domain, n_init_data)
data.y = [f(xi) for xi in data.x]

# Set acqfunction
acqfn_params = {"n_path": 15, "crop": True}
acqfn_class = MultiBaxAcqFunction
n_rand_acqopt = 500

# Compute true path
true_algo = algo_class(algo_params)
full_path, output = true_algo.run_algorithm_on_f(f)
true_path = true_algo.get_exe_path_crop()

# Plot
save_figure = True
fig, ax = plt.subplots(1, 1, figsize=(5, 5))


# Plot true path and posterior path samples
returns = []
path_lengths = []
for _ in trange(10):
    full_path, output = true_algo.run_algorithm_on_f(f)
    tp = true_algo.get_exe_path_crop()
    path_lengths.append(len(full_path.x))
    plot_path_2d(tp, ax)
    returns.append(compute_return(output[2], 1))
returns = np.array(returns)
path_lengths = np.array(path_lengths)
print(f"GT Results: returns.mean()={returns.mean()} returns.std()={returns.std()}")
print(
    f"GT Execution: path_lengths.mean()={path_lengths.mean()} path_lengths.std()={path_lengths.std()}"
)

# Plot settings
ax.set(
    xlim=(domain[0][0] - 0.1, domain[0][1] + 0.1),
    ylim=(domain[1][0] - 0.1, domain[1][1] + 0.1),
    xlabel="$v$",
    ylabel="$h$",
)

if save_figure:
    neatplot.save_figure(f"mpc_gt", "pdf")

# Run BAX loop
n_iter = 1

for i in range(n_iter):
    print("---" * 5 + f" Start iteration i={i} " + "---" * 5)

    # Set model
    model = gp_model_class(multi_gp_params, data)

    # Set and optimize acquisition function
    acqfn = acqfn_class(acqfn_params, model, algo)
    x_test = unif_random_sample_domain(domain, n=n_rand_acqopt)
    acqopt = AcqOptimizer({"x_batch": x_test})
    x_next, x_next_val = acqopt.optimize(acqfn)

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))

    # Plot observations
    x_obs = [xi[0] for xi in data.x]
    y_obs = [xi[1] for xi in data.x]
    ax.scatter(x_obs, y_obs, color="grey", s=5, alpha=0.1)  # small grey dots
    # ax.scatter(x_obs, y_obs, color='k', s=120)             # big black dots

    # Plot true path and posterior path samples
    plot_path_2d(true_path, ax, true_path=True)
    for path in acqfn.exe_path_list:
        plot_path_2d(path, ax)

    # Plot x_next
    ax.scatter(x_next[0], x_next[1], color="deeppink", s=120, zorder=100)

    # Plot settings
    ax.set(
        xlim=(domain[0][0] - 0.1, domain[0][1] + 0.1),
        ylim=(domain[1][0] - 0.1, domain[1][1] + 0.1),
        xlabel="$v$",
        ylabel="$h$",
    )

    if save_figure:
        neatplot.save_figure(f"mpc_{i}", "pdf")

    # Query function, update data
    print(f"Length of data.x: {len(data.x)}")
    print(f"Length of data.y: {len(data.y)}")
    y_next = f(x_next)
    data.x.append(x_next)
    data.y.append(y_next)
