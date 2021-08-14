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
import random
from matplotlib import pyplot as plt

from bax.models.gpfs_gp import BatchMultiGpfsGp
from bax.acq.acquisition import MultiBaxAcqFunction, MCAcqFunction
from bax.acq.acqoptimize import AcqOptimizer
from bax.alg.mpc import MPC
from bax.util.misc_util import Dumper
from bax.util import envs
from bax.util.envs.wrappers import NormalizedEnv, make_normalized_reward_function, make_update_obs_fn
from bax.util.control_util import get_f_batch_mpc, get_f_batch_mpc_reward, compute_return, evaluate_policy
from bax.util.control_util import rollout_mse, mse
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
    env = gym.make(config.env.name)
    env.seed(seed)
    obs_dim = env.observation_space.low.size
    action_dim = env.action_space.low.size

    plan_env = gym.make(config.env.name)
    plan_env.seed(seed)
    reward_function = envs.reward_functions[config.env.name] if not config.alg.learn_reward else None
    if config.normalize_env:
        env = NormalizedEnv(env)
        plan_env = NormalizedEnv(plan_env)
        if reward_function is not None:
            reward_function = make_normalized_reward_function(plan_env, reward_function)
    if config.alg.learn_reward:
        f = get_f_batch_mpc_reward(plan_env, use_info_delta=config.teleport)
    else:
        f = get_f_batch_mpc(plan_env, use_info_delta=config.teleport)
    update_fn = make_update_obs_fn(env, teleport=config.teleport)

    start_obs = env.reset() if config.fixed_start_obs else None
    logging.info(f"Start obs: {start_obs}")

    # Set domain
    low = np.concatenate([env.observation_space.low, env.action_space.low])
    high = np.concatenate([env.observation_space.high, env.action_space.high])
    domain = [elt for elt in zip(low, high)]

    # Set algorithm
    algo_class = MPC
    algo_params = dict(
            start_obs=start_obs,
            env=plan_env,
            reward_function=reward_function,
            project_to_domain=False,
            base_nsamps=config.mpc.nsamps,
            planning_horizon=config.mpc.planning_horizon,
            n_elites=config.mpc.n_elites,
            beta=config.mpc.beta,
            gamma=config.mpc.gamma,
            xi=config.mpc.xi,
            num_iters=config.mpc.num_iters,
            actions_per_plan=config.mpc.actions_per_plan,
            domain=domain,
            action_lower_bound=env.action_space.low,
            action_upper_bound=env.action_space.high,
            crop_to_domain=config.crop_to_domain,
            update_fn=update_fn,
    )
    algo = algo_class(algo_params)

    # Set data
    data = Namespace()
    data.x = unif_random_sample_domain(domain, config.num_init_data)
    data.y = f(data.x)

    # Make a test set for model evalution separate from the controller
    test_data = Namespace()
    test_data.x = unif_random_sample_domain(domain, config.test_set_size)
    test_data.y = f(test_data.x)

    # Set model
    gp_params = {'ls': config.env.gp.ls, 'alpha': config.env.gp.alpha, 'sigma': config.env.gp.sigma, 'n_dimx': obs_dim +
                 action_dim}
    multi_gp_params = {'n_dimy': obs_dim, 'gp_params': gp_params}
    gp_model_class = BatchMultiGpfsGp

    # Set acqfunction
    acqfn_params = {'n_path': config.n_paths, 'crop': True}
    acqfn_class = MultiBaxAcqFunction
    n_rand_acqopt = config.n_rand_acqopt

    # Compute true path and associated test set
    true_algo = algo_class(algo_params)
    full_path, output = true_algo.run_algorithm_on_f(f)
    true_path = true_algo.get_exe_path_crop()
    true_path_data = list(zip(true_algo.exe_path.x, true_algo.exe_path.y))
    test_points = random.sample(true_path_data, config.test_set_size)
    test_mpc_data = Namespace()
    test_mpc_data.x = [tp[0] for tp in test_points]
    test_mpc_data.y = [tp[1] for tp in test_points]

    # set plot fn
    plot_fn = plotters[config.env.name]

    ax = None
    # Compute and plot true path (on true function) multiple times
    returns = []
    path_lengths = []
    pbar = trange(config.num_eval_trials)
    for _ in pbar:
        full_path, output = true_algo.run_algorithm_on_f(f)
        tp = true_algo.get_exe_path_crop()
        path_lengths.append(len(full_path.x))
        ax = plot_fn(tp, ax, domain, 'true')
        returns.append(compute_return(output[2], 1))
        stats = {"Mean Return": np.mean(returns), "Std Return:": np.std(returns)}
        pbar.set_postfix(stats)
    returns = np.array(returns)
    dumper.add('GT Returns', returns)
    path_lengths = np.array(path_lengths)
    logging.info(f"GT Results: returns.mean()={returns.mean()} returns.std()={returns.std()}")
    logging.info(f"GT Execution: path_lengths.mean()={path_lengths.mean()} path_lengths.std()={path_lengths.std()}")
    neatplot.save_figure(str(dumper.expdir / 'mpc_gt'), 'png')

    for i in range(config.num_iters):
        logging.info('---' * 5 + f' Start iteration i={i} ' + '---' * 5)
        logging.info(f'Length of data.x: {len(data.x)}')
        logging.info(f'Length of data.y: {len(data.y)}')
        ax = None

        # Set model
        model = gp_model_class(multi_gp_params, data)

        if config.alg.use_acquisition:
            # Set and optimize acquisition function
            acqfn_base = acqfn_class(acqfn_params, model, algo)
            acqfn = MCAcqFunction(acqfn_base, {"num_samples_mc": config.num_samples_mc})
            x_test = unif_random_sample_domain(domain, n=n_rand_acqopt)
            acqopt = AcqOptimizer({"x_batch": x_test})
            x_next, acq_val = acqopt.optimize(acqfn)
            dumper.add('Acquisition Function Value', acq_val)

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
        if i % config.eval_frequency == 0 or i + 1 == config.num_iters or config.alg.rollout_all:
            with Timer("Evaluate the current MPC policy"):
                # execute the best we can
                # this is required to delete the current execution path
                algo.initialize()

                def postmean_fn(x):
                    mu_list, std_list = model.get_post_mu_cov(x, full_cov=False)
                    mu_tup_for_x = list(zip(*mu_list))
                    return mu_tup_for_x
                policy = partial(algo.execute_mpc, f=postmean_fn)
                real_returns = []
                mses = []
                pbar = trange(config.num_eval_trials)
                for j in pbar:
                    real_obs, real_actions, real_rewards = evaluate_policy(env, policy, start_obs=start_obs,
                                                                           mpc_pass=True)
                    real_return = compute_return(real_rewards, 1)
                    real_returns.append(real_return)
                    real_path_mpc = Namespace()
                    real_path_mpc.x = real_obs
                    ax = plot_fn(real_path_mpc, ax, domain, 'postmean')
                    mses.append(rollout_mse(algo.old_exe_paths[-1], f))
                    stats = {"Mean Return": np.mean(real_returns), "Std Return:": np.std(real_returns),
                             "Model MSE": np.mean(mses)}
                    pbar.set_postfix(stats)
                real_returns = np.array(real_returns)
                algo.old_exe_paths = []
                dumper.add('Eval Returns', real_returns)
                dumper.add('Eval ndata', len(data.x))
                logging.info(f"Eval Results: real_returns={real_returns}")
                current_mpc_mse = np.mean(mses)
                test_y_hat = postmean_fn(test_data.x)
                random_mse = mse(test_data.y, test_y_hat)
                gt_mpc_y_hat = postmean_fn(test_mpc_data.x)
                gt_mpc_mse = mse(test_mpc_data.y, gt_mpc_y_hat)
                dumper.add('Model MSE (current MPC)', current_mpc_mse)
                dumper.add('Model MSE (random test set)', random_mse)
                dumper.add('Model MSE (GT MPC)', gt_mpc_mse)
                logging.info(f"Current MPC MSE: {current_mpc_mse:.3f}")
                logging.info(f"Random MSE: {random_mse:.3f}")
                logging.info(f"GT MPC MSE: {gt_mpc_mse:.3f}")

            save_figure = True
        if save_figure:
            neatplot.save_figure(str(dumper.expdir / f'mpc_{i}'), 'png')
        dumper.save()

        # Query function, update data
        if config.alg.use_rollout_data:
            new_x = [np.concatenate((obs, action)) for obs, action in zip(real_obs, real_actions)]
            new_y = f(new_x)

            data.x.extend(new_x)
            data.y.extend(new_y)
        else:
            y_next = f([x_next])[0]
            data.x.append(x_next)
            data.y.append(y_next)
        plt.close('all')


if __name__ == '__main__':
    main()
