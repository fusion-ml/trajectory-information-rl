"""
Testing MultiGpfsGp and MultiBaxAcqFunction classes
"""

from argparse import Namespace
import logging
import numpy as np
import gym
from tqdm import trange
from functools import partial
import tensorflow as tf
import hydra

from bax.models.gpfs_gp import MultiGpfsGp
from bax.acq.acquisition import MultiBaxAcqFunction
from bax.acq.acqoptimize import AcqOptimizer
from bax.alg.mpc import MPC
from bax.util.misc_util import Dumper
from bax.util import envs
from bax.util.control_util import get_f_mpc, get_f_mpc_reward, compute_return, evaluate_policy, rollout_mse
from bax.util.domain_util import unif_random_sample_domain
from bax.util.timing import Timer
from bax.viz import plotters
import neatplot


@hydra.main(config_path='cfg', config_name='config')
def main(config):
    dumper = Dumper(config.name)

    # Set plot settings
    neatplot.set_style()
    neatplot.update_rc('figure.dpi', 120)
    neatplot.update_rc('text.usetex', False)

    # Set random seed
    seed = config.seed
    np.random.seed(seed)
    tf.random.set_seed(seed)

    assert (not config.fixed_start_obs) or config.num_samples_mc == 1, f"Need to have a fixed start obs ({config.fixed_start_obs}) or only 1 mc sample ({config.num_samples_mc})"  # NOQA

    # -------------
    # Start Script
    # -------------
    # Set black-box function
    env = gym.make(config.env)
    env.seed(seed)
    obs_dim = env.observation_space.low.size
    action_dim = env.action_space.low.size

    plan_env = gym.make(config.env.name)
    plan_env.seed(seed)
    f = get_f_mpc(plan_env) if not config.alg.learn_reward else get_f_mpc_reward(plan_env)
    start_obs = env.reset() if config.fixed_start_obs else None

    # Set domain
    low = np.concatenate([env.observation_space.low, env.action_space.low])
    high = np.concatenate([env.observation_space.high, env.action_space.high])
    domain = [elt for elt in zip(low, high)]

    # Set algorithm
    algo_class = MPC
    algo_params = dict(
            start_obs=start_obs,
            env=plan_env,
            reward_function=envs.reward_functions[config.env.name] if not config.alg.learn_reward else None,
            project_to_domain=True,
            base_nsamps=config.mpc.nsamps,
            planning_horizon=config.mpc.planning_horizon,
            n_elites=config.mpc.n_elites,
            beta=config.mpc.beta,
            gamma=config.mpc.gamma,
            xi=config.mpc.xi,
            num_iters=config.mpc.num_iters,
            actions_per_plan=config.mpc.actions_per_plan,
            domain=domain,
    )
    algo = algo_class(algo_params)

    # Set data
    data = Namespace()
    n_init_data = 1
    data.x = unif_random_sample_domain(domain, n_init_data)
    data.y = [f(xi) for xi in data.x]

    # Set model
    gp_params = {'ls': config.env.gp.ls, 'alpha': config.env.alpha, 'sigma': config.env.sigma, 'n_dimx': obs_dim +
                 action_dim}
    multi_gp_params = {'n_dimy': obs_dim, 'gp_params': gp_params}
    gp_model_class = MultiGpfsGp

    # Set acqfunction
    acqfn_params = {'n_path': 15, 'crop': True}
    acqfn_class = MultiBaxAcqFunction
    n_rand_acqopt = 500

    # Compute true path
    true_algo = algo_class(algo_params)
    full_path, output = true_algo.run_algorithm_on_f(f)
    true_path = true_algo.get_exe_path_crop()

    # set plot fn
    plot_fn = plotters[config.env.name]

    ax = None
    # Compute and plot true path (on true function) multiple times
    returns = []
    path_lengths = []
    for _ in trange(10):
        full_path, output = true_algo.run_algorithm_on_f(f)
        tp = true_algo.get_exe_path_crop()
        path_lengths.append(len(full_path.x))
        ax = plot_fn(tp, ax, domain, 'true')
        returns.append(compute_return(output[2], 1))
    returns = np.array(returns)
    path_lengths = np.array(path_lengths)
    logging.info(f"GT Results: returns.mean()={returns.mean()} returns.std()={returns.std()}")
    logging.info(f"GT Execution: path_lengths.mean()={path_lengths.mean()} path_lengths.std()={path_lengths.std()}")
    neatplot.save_figure(str(dumper.expdir / 'mpc_gt'), 'pdf')

    for i in range(config.num_iters):
        logging.info('---' * 5 + f' Start iteration i={i} ' + '---' * 5)
        logging.info(f'Length of data.x: {len(data.x)}')
        logging.info(f'Length of data.y: {len(data.y)}')
        ax = None

        # Set model
        model = gp_model_class(multi_gp_params, data)

        if not config.alg.mbrl:
            # Set and optimize acquisition function
            acqfn = acqfn_class(acqfn_params, model, algo)
            x_test = unif_random_sample_domain(domain, n=n_rand_acqopt)
            acqopt = AcqOptimizer({"x_batch": x_test, "num_samples_mc": config.num_samples_mc})
            x_next = acqopt.optimize(acqfn)

            # Plot true path and posterior path samples
            ax = plot_fn(true_path, ax, domain, 'true')
            # Plot observations
            x_obs = [xi[0] for xi in data.x]
            y_obs = [xi[1] for xi in data.x]
            ax.scatter(x_obs, y_obs, color='grey', s=5, alpha=0.1)  # small grey dots
            # ax.scatter(x_obs, y_obs, color='k', s=120)             # big black dots

            for path in acqfn.exe_path_list:
                ax = plot_fn(path, ax, domain, 'samp')

            # Plot x_next
            ax.scatter(x_next[0], x_next[1], color='deeppink', s=120, zorder=100)

        save_figure = False
        if i % config.eval_frequency == 0 or i + 1 == config.num_iters or config.alg.mbrl:
            with Timer("Evaluate the current MPC policy"):
                # execute the best we can
                # this is required to delete the current execution path
                algo.initialize()

                def postmean_fn(x):
                    mu_list, std_list = model.get_post_mu_cov([x], full_cov=False)
                    mu_tup_for_x = list(zip(*mu_list))[0]
                    return mu_tup_for_x
                policy = partial(algo.execute_mpc, f=postmean_fn)
                real_returns = []
                for j in range(config.num_eval_trials):
                    real_obs, real_actions, real_rewards = evaluate_policy(env, policy, start_obs=start_obs,
                                                                           mpc_pass=True)
                    real_return = compute_return(real_rewards, 1)
                    real_returns.append(real_return)
                    real_path_mpc = Namespace()
                    real_path_mpc.x = real_obs
                    ax = plot_fn(real_path_mpc, ax, domain, 'postmean')
                real_returns = np.array(real_returns)
                old_exe_paths = algo.old_exe_paths
                algo.old_exe_paths = []
                logging.info(f"Return on executed MPC: {np.mean(real_returns)}, std: {np.std(real_returns)}")
                dumper.add('Eval Returns', real_returns)
                dumper.add('Eval ndata', len(data.x))
                mse = np.mean([rollout_mse(path, f) for path in old_exe_paths])
                logging.info(f"Model MSE during test time rollout: {mse}")
                dumper.add('Model MSE', mse)

            save_figure = True
        if save_figure:
            neatplot.save_figure(str(dumper.expdir / f'mpc_{i}'), 'pdf')
        dumper.save()

        # Query function, update data
        if config.alg.mbrl:
            new_x = [np.concatenate((obs, action)) for obs, action in zip(real_obs, real_actions)]
            new_y = [f(x) for x in new_x]

            data.x.extend(new_x)
            data.y.extend(new_y)
        else:
            y_next = f(x_next)
            data.x.append(x_next)
            data.y.append(y_next)


if __name__ == '__main__':
    main()
