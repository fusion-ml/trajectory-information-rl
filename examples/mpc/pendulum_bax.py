"""
Testing MultiGpfsGp and MultiBaxAcqFunction classes
"""

from argparse import Namespace
import argparse
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
from functools import partial
import tensorflow as tf

from bax.models.gpfs_gp import MultiGpfsGp
from bax.acq.acquisition import MultiBaxAcqFunction
from bax.acq.acqoptimize import AcqOptimizer
from bax.alg.mpc import MPC
from bax.util.misc_util import Dumper
from bax.util.envs.pendulum import PendulumEnv, pendulum_reward
from bax.util.control_util import (
    get_f_batch_mpc,
    get_f_batch_mpc_reward,
    compute_return,
    evaluate_policy,
    rollout_mse,
)
from bax.util.domain_util import unif_random_sample_domain
from bax.util.timing import Timer
import neatplot


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("name", help="The name of the experiment and output directory.")
    parser.add_argument("-ow", dest="overwrite", action="store_true")
    parser.add_argument("-net", "--num_eval_trials", type=int, default=1)
    parser.add_argument("--eval_frequency", type=int, default=25)
    parser.add_argument("-app", "--actions_per_plan", type=int, default=6)
    parser.add_argument("-ni", "--n_iter", type=int, default=200)
    parser.add_argument("-s", "--seed", type=int, default=11)
    parser.add_argument("-lr", "--learn_reward", action="store_true")
    parser.add_argument("-nms", "--num_mean_samples", type=int, default=100)
    parser.add_argument(
        "-epm",
        "--exact_postmean",
        action="store_true",
        help="Use the exact posterior mean at test time instead of sampling with -nms",
    )
    return parser.parse_args()


args = parse_arguments()
dumper = Dumper(args.name, args, args.overwrite)

# Set plot settings
neatplot.set_style()
neatplot.update_rc("figure.dpi", 120)
neatplot.update_rc("text.usetex", False)


# Set random seed
seed = args.seed
np.random.seed(seed)
tf.random.set_seed(seed)


def plot_path_2d(path, ax=None, path_str="samp"):
    """Plot a path through an assumed two-dimensional state space."""
    assert path_str in ["samp", "true", "postmean"]

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))

    x_plot = [xi[0] for xi in path.x]
    y_plot = [xi[1] for xi in path.x]

    if path_str == "true":
        ax.plot(x_plot, y_plot, "k--", linewidth=3)
        ax.plot(x_plot, y_plot, "*", color="k", markersize=5)
    elif path_str == "postmean":
        ax.plot(x_plot, y_plot, "r--", linewidth=3)
        ax.plot(x_plot, y_plot, "*", color="r", markersize=5)
    elif path_str == "samp":
        ax.plot(x_plot, y_plot, "k--", linewidth=1, alpha=0.3)
        ax.plot(x_plot, y_plot, "o", alpha=0.3)


# -------------
# Start Script
# -------------
# Set black-box function
env = PendulumEnv(seed=seed)
obs_dim = env.observation_space.low.size
action_dim = env.action_space.low.size

plan_env = PendulumEnv(seed=seed)
f = (
    get_f_batch_mpc(plan_env)
    if not args.learn_reward
    else get_f_batch_mpc_reward(plan_env)
)
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
    reward_function=pendulum_reward if not args.learn_reward else None,
    project_to_domain=True,
    base_nsamps=10,
    planning_horizon=20,
    n_elites=3,
    beta=3,
    gamma=1.25,
    xi=0.3,
    num_iters=3,
    actions_per_plan=args.actions_per_plan,
    domain=domain,
)
algo = algo_class(algo_params)

# Set data
data = Namespace()
n_init_data = 1
data.x = unif_random_sample_domain(domain, n_init_data)
data.y = f(data.x)

# Set model
gp_params = {"ls": 0.85, "alpha": 1.0, "sigma": 1e-2, "n_dimx": obs_dim + action_dim}
multi_gp_params = {"n_dimy": obs_dim, "gp_params": gp_params}
gp_model_class = MultiGpfsGp

# Set acqfunction
acqfn_params = {"n_path": 15, "crop": True}
acqfn_class = MultiBaxAcqFunction
n_rand_acqopt = 500

# Compute true path
true_algo = algo_class(algo_params)
full_path, output = true_algo.run_algorithm_on_f(f)
true_path = true_algo.get_exe_path_crop()

# Plot
fig, ax = plt.subplots(1, 1, figsize=(5, 5))


# Compute and plot true path (on true function) multiple times
returns = []
path_lengths = []
for _ in trange(10):
    full_path, output = true_algo.run_algorithm_on_f(f)
    tp = true_algo.get_exe_path_crop()
    path_lengths.append(len(full_path.x))
    plot_path_2d(tp, ax, "true")
    returns.append(compute_return(output[2], 1))
returns = np.array(returns)
path_lengths = np.array(path_lengths)
print(f"GT Results: returns.mean()={returns.mean()} returns.std()={returns.std()}")
print(
    f"GT Execution: path_lengths.mean()={path_lengths.mean()} path_lengths.std()={path_lengths.std()}"
)

# Plot settings
ax.set(
    xlim=(domain[0][0], domain[0][1]),
    ylim=(domain[1][0], domain[1][1]),
    xlabel="$x$",
    ylabel="$\\theta$",
)

save_figure = True
if save_figure:
    neatplot.save_figure(str(dumper.expdir / "mpc_gt"), "pdf")


for i in range(args.n_iter):
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
    plot_path_2d(true_path, ax, "true")
    for path in acqfn.exe_path_list:
        plot_path_2d(path, ax, "samp")

    # Plot x_next
    ax.scatter(x_next[0], x_next[1], color="deeppink", s=120, zorder=100)

    # Plot settings
    ax.set(
        xlim=(domain[0][0], domain[0][1]),
        ylim=(domain[1][0], domain[1][1]),
        xlabel="$x$",
        ylabel="$\\theta$",
    )

    # Query function, update data
    print(f"Length of data.x: {len(data.x)}")
    print(f"Length of data.y: {len(data.y)}")

    save_figure = False
    if i % args.eval_frequency == 0 or i + 1 == args.n_iter:
        with Timer("Evaluate the current MPC policy"):
            # execute the best we can
            # this is required to delete the current execution path
            algo.initialize()
            # TODO: get this to work on the mean
            if args.exact_postmean:

                def postmean_fn(x_list):
                    mu_list, std_list = model.get_post_mu_cov([x_list], full_cov=False)
                    mu_tup_for_x = list(zip(*mu_list))[0]
                    return mu_tup_for_x

            else:
                n_postmean_f_samp = args.num_mean_samples
                model.initialize_function_sample_list(n_postmean_f_samp)
                postmean_fn = model.call_function_sample_list_mean
            policy = partial(algo.execute_mpc, f=postmean_fn)
            real_returns = []
            for j in range(args.num_eval_trials):
                real_obs, real_actions, real_rewards = evaluate_policy(
                    env, policy, start_obs=start_obs, mpc_pass=True
                )
                real_return = compute_return(real_rewards, 1)
                real_returns.append(real_return)
                real_path_mpc = Namespace()
                real_path_mpc.x = real_obs
                plot_path_2d(real_path_mpc, ax, "postmean")
            real_returns = np.array(real_returns)
            old_exe_paths = algo.old_exe_paths
            algo.old_exe_paths = []
            print(
                f"Return on executed MPC: {np.mean(real_returns)}, std: {np.std(real_returns)}"
            )
            dumper.add("Eval Returns", real_returns)
            dumper.add("Eval ndata", len(data.x))
            mse = np.mean([rollout_mse(path, f) for path in old_exe_paths])
            print(f"Model MSE during test time rollout: {mse}")
            dumper.add("Model MSE", mse)

        save_figure = True
    if save_figure:
        neatplot.save_figure(str(dumper.expdir / f"mpc_{i}"), "pdf")
    dumper.save()

    y_next = f(x_next)
    data.x.append(x_next)
    data.y.append(y_next)
