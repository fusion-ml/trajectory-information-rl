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
from bax.models.gpflow_gp import get_gpflow_hypers_from_data
from bax.acq.acquisition import MultiBaxAcqFunction, MCAcqFunction, UncertaintySamplingAcqFunction
from bax.acq.acqoptimize import AcqOptimizer
from bax.alg.mpc import MPC
from bax import envs
from bax.envs.wrappers import NormalizedEnv, make_normalized_reward_function, make_update_obs_fn
from bax.envs.wrappers import make_normalized_plot_fn
from bax.util.misc_util import Dumper, make_postmean_fn
from bax.util.control_util import get_f_batch_mpc, get_f_batch_mpc_reward, compute_return, evaluate_policy
from bax.util.control_util import rollout_mse, mse
from bax.util.domain_util import unif_random_sample_domain
from bax.util.timing import Timer
from bax.viz import plotters, make_plot_obs
import neatplot

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


@hydra.main(config_path='cfg', config_name='config')
def main(config):
    # ==============================================
    #   Define and configure
    # ==============================================
    dumper = Dumper(config.name)

    # Set plot settings
    neatplot.set_style()
    neatplot.update_rc('figure.dpi', 120)
    neatplot.update_rc('text.usetex', False)

    # Set random seed
    seed = config.seed
    np.random.seed(seed)
    tf.random.set_seed(seed)

    # Check fixed_start_obs and num_samples_mc compatability
    assert (not config.fixed_start_obs) or config.num_samples_mc == 1, f"Need to have a fixed start obs ({config.fixed_start_obs}) or only 1 mc sample ({config.num_samples_mc})"  # NOQA

    # Set black-box functions
    env = gym.make(config.env.name)
    env.seed(seed)
    obs_dim = env.observation_space.low.size
    action_dim = env.action_space.low.size

    plan_env = gym.make(config.env.name)
    plan_env.seed(seed)

    # set plot fn
    plot_fn = partial(plotters[config.env.name], env=plan_env)

    reward_function = envs.reward_functions[config.env.name] if not config.alg.learn_reward else None
    if config.normalize_env:
        env = NormalizedEnv(env)
        plan_env = NormalizedEnv(plan_env)
        if reward_function is not None:
            reward_function = make_normalized_reward_function(plan_env, reward_function)
        plot_fn = make_normalized_plot_fn(plan_env, plot_fn)
    if config.alg.learn_reward:
        f = get_f_batch_mpc_reward(plan_env, use_info_delta=config.teleport)
    else:
        f = get_f_batch_mpc(plan_env, use_info_delta=config.teleport)
    update_fn = make_update_obs_fn(env, teleport=config.teleport)

    # Set start obs
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

    # Set initial data
    data = Namespace()
    data.x = unif_random_sample_domain(domain, config.num_init_data)
    data.y = f(data.x)
    for x, y in zip(data.x, data.y):
        dumper.add('x', data.x)
        dumper.add('y', data.y)

    # Plot initial data
    ax_obs_init, fig_obs_init = plot_fn(path=None, domain=domain)
    x_obs = [xi[0] for xi in data.x]
    y_obs = [xi[1] for xi in data.x]
    if ax_obs_init:
        ax_obs_init.plot(x_obs, y_obs, 'o', color='k', ms=1)
        neatplot.save_figure(str(dumper.expdir / 'mpc_obs_init'), 'png', fig=fig_obs_init)

    # Make a test set for model evalution separate from the controller
    test_data = Namespace()
    test_data.x = unif_random_sample_domain(domain, config.test_set_size)
    test_data.y = f(test_data.x)

    # Set model
    gp_params = {
        'ls': config.env.gp.ls,
        'alpha': config.env.gp.alpha,
        'sigma': config.env.gp.sigma,
        'n_dimx': obs_dim + action_dim
    }
    if config.env.gp.periodic:
        gp_params['kernel_str'] = 'rbf_periodic'
        gp_params['periodic_dims'] = env.periodic_dimensions
        gp_params['period'] = config.env.gp.period
    multi_gp_params = {'n_dimy': obs_dim, 'gp_params': gp_params}
    gp_model_class = BatchMultiGpfsGp

    # Set acqfunction
    acqfn_params = {'n_path': config.n_paths, 'crop': True}
    acqfn_class = MultiBaxAcqFunction if not config.alg.uncertainty_sampling else UncertaintySamplingAcqFunction

    # ==============================================
    #   Computing groundtruth trajectories
    # ==============================================
    # Instantiate true algo and axes/figures
    true_algo = algo_class(algo_params)
    ax_gt, fig_gt = None, None

    # Compute and plot true path (on true function) multiple times
    full_paths = []
    true_paths = []
    returns = []
    path_lengths = []
    test_mpc_data = Namespace(x=[], y=[])
    pbar = trange(config.num_eval_trials)
    for i in pbar:
        # Run algorithm and extract paths
        full_path, output = true_algo.run_algorithm_on_f(f)
        full_paths.append(full_path)
        path_lengths.append(len(full_path.x))
        true_path = true_algo.get_exe_path_crop()
        true_paths.append(true_path)

        # Extract fraction of planning data for test_mpc_data
        true_planning_data = list(zip(true_algo.exe_path.x, true_algo.exe_path.y))
        test_points = random.sample(true_planning_data, int(config.test_set_size/config.num_eval_trials))
        new_x = [test_pt[0] for test_pt in test_points]
        new_y = [test_pt[1] for test_pt in test_points]
        test_mpc_data.x.extend(new_x)
        test_mpc_data.y.extend(new_y)

        # Plot groundtruth paths and print info
        ax_gt, fig_gt = plot_fn(true_path, ax_gt, fig_gt, domain, 'samp')
        returns.append(compute_return(output[2], 1))
        stats = {"Mean Return": np.mean(returns), "Std Return:": np.std(returns)}
        pbar.set_postfix(stats)

    # Log and dump
    returns = np.array(returns)
    dumper.add('GT Returns', returns)
    path_lengths = np.array(path_lengths)
    logging.info(f"GT Returns: returns{returns}")
    logging.info(f"GT Returns: mean={returns.mean()} std={returns.std()}")
    logging.info(f"GT Execution: path_lengths.mean()={path_lengths.mean()} path_lengths.std()={path_lengths.std()}")
    all_x = []
    for fp in full_paths:
        all_x += fp.x
    all_x = np.array(all_x)
    print(f"all_x.shape = {all_x.shape}")
    print(f"all_x.min(axis=0) = {all_x.min(axis=0)}")
    print(f"all_x.max(axis=0) = {all_x.max(axis=0)}")
    print(f"all_x.mean(axis=0) = {all_x.mean(axis=0)}")
    print(f"all_x.var(axis=0) = {all_x.var(axis=0)}")

    # Save groundtruth paths plot
    if fig_gt:
        neatplot.save_figure(str(dumper.expdir / 'mpc_gt'), 'png', fig=fig_gt)

    # ==============================================
    #   Optionally: fit GP hyperparameters (then exit)
    # ==============================================
    if config.fit_hypers:
        # Use test_mpc_data to fit hyperparameters
        fit_data = test_mpc_data
        assert len(fit_data.x) <= 3000, "fit_data larger than preset limit (can cause memory issues)"

        logging.info('\n'+'='*60+'\n Fitting Hyperparameters\n'+'='*60)
        logging.info(f'Number of observations in fit_data: {len(fit_data.x)}')

        # Plot hyper fitting data
        ax_obs_hyper_fit, fig_obs_hyper_fit = plot_fn(path=None, domain=domain)
        x_obs = [xi[0] for xi in fit_data.x]
        y_obs = [xi[1] for xi in fit_data.x]
        if ax_obs_hyper_fit:
            ax_obs_hyper_fit.plot(x_obs, y_obs, 'o', color='k', ms=1)
            neatplot.save_figure(str(dumper.expdir / 'mpc_obs_hyper_fit'), 'png', fig=fig_obs_hyper_fit)

        # Perform hyper fitting
        for idx in range(len(data.y[0])):
            data_fit = Namespace(x=fit_data.x, y=[yi[idx] for yi in fit_data.y])
            gp_params = get_gpflow_hypers_from_data(data_fit, print_fit_hypers=True,
                                                    opt_max_iter=config.env.gp.opt_max_iter)
            logging.info(f'gp_params for output {idx} = {gp_params}')

        # End script if hyper fitting bc need to include in config
        return

    # ==============================================
    #   Run main algorithm loop
    # ==============================================

    # Set current_obs as fixed start_obs or reset plan_env
    if config.alg.rollout_sampling:
        current_obs = start_obs.copy() if config.fixed_start_obs else plan_env.reset()
        current_t = 0

    posterior_returns = None

    for i in range(config.num_iters):
        logging.info('---' * 5 + f' Start iteration i={i} ' + '---' * 5)
        logging.info(f'Length of data.x: {len(data.x)}')
        logging.info(f'Length of data.y: {len(data.y)}')

        # Initialize various axes and figures
        ax_all, fig_all = plot_fn(path=None, domain=domain)
        ax_postmean, fig_postmean = plot_fn(path=None, domain=domain)
        ax_samp, fig_samp = plot_fn(path=None, domain=domain)
        ax_obs, fig_obs = plot_fn(path=None, domain=domain)

        # Set model as None, instantiate when needed
        model = None

        if config.alg.use_acquisition:
            model = gp_model_class(multi_gp_params, data)
            # Set and optimize acquisition function
            acqfn_base = acqfn_class(params=acqfn_params, model=model, algorithm=algo)
            acqfn = MCAcqFunction(acqfn_base, {"num_samples_mc": config.num_samples_mc})
            acqopt = AcqOptimizer()
            acqopt.initialize(acqfn)
            if config.alg.rollout_sampling:
                x_test = [np.concatenate([current_obs, env.action_space.sample()]) for _ in range(config.n_rand_acqopt)]
            elif config.sample_exe and not config.alg.uncertainty_sampling:
                all_x = []
                for path in acqfn.exe_path_full_list:
                    all_x += path.x
                n_path = int(config.n_rand_acqopt * config.path_sampling_fraction)
                n_rand = config.n_rand_acqopt - n_path
                x_test = random.sample(all_x, n_path)
                x_test = np.array(x_test)
                x_test += np.random.randn(*x_test.shape) * 0.01
                x_test = list(x_test)
                x_test += unif_random_sample_domain(domain, n=n_rand)
            else:
                x_test = unif_random_sample_domain(domain, n=config.n_rand_acqopt)
            x_next, acq_val = acqopt.optimize(x_test)
            dumper.add('Acquisition Function Value', acq_val)
            dumper.add('x_next', x_next)

            # Plot true path and posterior path samples
            ax_all, fig_all = plot_fn(true_path, ax_all, fig_all, domain, 'true')
            if ax_all is not None:
                # Plot observations
                x_obs, y_obs = make_plot_obs(data.x, env, config.env.normalize_env)
                ax_all.scatter(x_obs, y_obs, color='grey', s=10, alpha=0.3)
                ax_obs.plot(x_obs, y_obs, 'o', color='k', ms=1)

                # Plot execution path posterior samples
                for path in acqfn.exe_path_list:
                    ax_all, fig_all = plot_fn(path, ax_all, fig_all, domain, 'samp')
                    ax_samp, fig_samp = plot_fn(path, ax_samp, fig_samp, domain, 'samp')

                # Plot x_next
                x, y = make_plot_obs(x_next, env, config.env.normalize_env)
                ax_all.scatter(x, y, facecolors='deeppink', edgecolors='k', s=120, zorder=100)
                ax_obs.plot(x, x, 'o', mfc='deeppink', mec='k', ms=12, zorder=100)

            # Store returns of posterior samples
            posterior_returns = [compute_return(output[2], 1) for output in acqfn.output_list]
            dumper.add('Posterior Returns', posterior_returns)
        elif config.alg.use_mpc:
            model = gp_model_class(multi_gp_params, data)
            algo.initialize()

            policy = partial(algo.execute_mpc, f=make_postmean_fn(model))
            action = policy(current_obs)
            x_next = np.concatenate([current_obs, action])
        else:
            x_next = unif_random_sample_domain(domain, 1)[0]

        # ==============================================
        #   Periodically run evaluation and plot
        # ==============================================
        if i % config.eval_frequency == 0 or i + 1 == config.num_iters:
            if model is None:
                model = gp_model_class(multi_gp_params, data)
            if posterior_returns:
                logging.info(f"Current posterior returns: {posterior_returns}")
                logging.info(f"Current posterior returns: mean={np.mean(posterior_returns)}, std={np.std(posterior_returns)}")
            with Timer("Evaluate the current MPC policy"):
                # execute the best we can
                # this is required to delete the current execution path
                algo.initialize()

                postmean_fn = make_postmean_fn(model)
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
                    ax_all, fig_all = plot_fn(real_path_mpc, ax_all, fig_all, domain, 'postmean')
                    ax_postmean, fig_postmean = plot_fn(real_path_mpc, ax_postmean, fig_postmean, domain, 'samp')
                    mses.append(rollout_mse(algo.old_exe_paths[-1], f))
                    stats = {"Mean Return": np.mean(real_returns), "Std Return:": np.std(real_returns),
                             "Model MSE": np.mean(mses)}
                    pbar.set_postfix(stats)
                real_returns = np.array(real_returns)
                algo.old_exe_paths = []
                dumper.add('Eval Returns', real_returns)
                dumper.add('Eval ndata', len(data.x))
                logging.info(f"Eval Results: real_returns={real_returns}")
                logging.info(f"Eval Results: mean={np.mean(real_returns)}, std={np.std(real_returns)}")
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

            # Save figure at end of evaluation
            neatplot.save_figure(str(dumper.expdir / f'mpc_all_{i}'), 'png', fig=fig_all)
            neatplot.save_figure(str(dumper.expdir / f'mpc_postmean_{i}'), 'png', fig=fig_postmean)
            neatplot.save_figure(str(dumper.expdir / f'mpc_samp_{i}'), 'png', fig=fig_samp)
            neatplot.save_figure(str(dumper.expdir / f'mpc_obs_{i}'), 'png', fig=fig_obs)


        # Query function, update data
        y_next = f([x_next])[0]
        data.x.append(x_next)
        data.y.append(y_next)
        dumper.add('x', x_next)
        dumper.add('y', y_next)
        if config.alg.rollout_sampling:
            current_t += 1
            if current_t > env.horizon:
                current_t = 0
                current_obs = start_obs.copy() if config.fixed_start_obs else plan_env.reset()
            else:
                current_obs += y_next[-obs_dim:]
        # Dumper save
        dumper.save()
        plt.close('all')


if __name__ == '__main__':
    main()
