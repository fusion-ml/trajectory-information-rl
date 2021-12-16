"""
Main File for BARL and associated code.
"""
from argparse import Namespace
import logging
import numpy as np
import gym
from tqdm import trange
from functools import partial
from copy import deepcopy
import tensorflow as tf
import hydra
import random
from matplotlib import pyplot as plt

from barl.models.gpfs_gp import BatchMultiGpfsGp, TFMultiGpfsGp
from barl.models.gpflow_gp import get_gpflow_hypers_from_data
from barl.acq.acquisition import MultiBaxAcqFunction, MCAcqFunction, UncertaintySamplingAcqFunction, KGRLAcqFunction, PILCOAcqFunction
from barl.acq.acqoptimize import AcqOptimizer, KGAcqOptimizer
from barl.alg.mpc import MPC
from barl import envs
from barl.envs.wrappers import NormalizedEnv, make_normalized_reward_function, make_update_obs_fn
from barl.envs.wrappers import make_normalized_plot_fn
from barl.util.misc_util import Dumper, make_postmean_fn, mse, model_likelihood
from barl.util.control_util import get_f_batch_mpc, get_f_batch_mpc_reward, compute_return, evaluate_policy
from barl.util.domain_util import unif_random_sample_domain
from barl.util.timing import Timer
from barl.viz import plotters, make_plot_obs, scatter, plot
import neatplot

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


@hydra.main(config_path='cfg', config_name='config')
def main(config):
    # ==============================================
    #   Define and configure
    # ==============================================
    dumper = Dumper(config.name)
    configure(config)

    # Instantiate environment and create functions for dynamics, plotting, rewards, state updates
    env, f, plot_fn, reward_function, update_fn, p0 = get_env(config)
    obs_dim = env.observation_space.low.size
    action_dim = env.action_space.low.size

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
            env=env,
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
    test_algo_params = dict(
            start_obs=start_obs,
            env=env,
            reward_function=reward_function,
            project_to_domain=False,
            base_nsamps=config.test_mpc.nsamps,
            planning_horizon=config.test_mpc.planning_horizon,
            n_elites=config.test_mpc.n_elites,
            beta=config.test_mpc.beta,
            gamma=config.test_mpc.gamma,
            xi=config.test_mpc.xi,
            num_iters=config.test_mpc.num_iters,
            actions_per_plan=config.test_mpc.actions_per_plan,
            domain=domain,
            action_lower_bound=env.action_space.low,
            action_upper_bound=env.action_space.high,
            crop_to_domain=config.crop_to_domain,
            update_fn=update_fn,
    )
    algo = algo_class(algo_params)
    test_algo = algo_class(test_algo_params)

    # Set initial data
    data = get_initial_data(config, env, f, domain, dumper, plot_fn)

    # Make a test set for model evalution separate from the controller
    test_data = Namespace()
    test_data.x = unif_random_sample_domain(domain, config.test_set_size)
    test_data.y = f(test_data.x)

    # Set model
    gp_model_class, gp_model_params = get_model(config, env, obs_dim, action_dim)

    # Set acqfunction
    acqfn_class, acqfn_params = get_acq_fn(config, env.horizon, p0, reward_function, update_fn,
                                           obs_dim, action_dim, gp_model_class, gp_model_params)
    acqopt_class, acqopt_params = get_acq_opt(config, obs_dim, action_dim, env, start_obs)

    # ==============================================
    #   Computing groundtruth trajectories
    # ==============================================
    true_path, test_mpc_data = execute_gt_mpc(config, algo_class, algo_params, f, dumper, domain, plot_fn)

    # ==============================================
    #   Optionally: fit GP hyperparameters (then exit)
    # ==============================================
    if config.fit_hypers:
        fit_hypers(config, test_mpc_data, plot_fn, domain, dumper.expdir)
        # End script if hyper fitting bc need to include in config
        return

    # ==============================================
    #   Run main algorithm loop
    # ==============================================

    # Set current_obs as fixed start_obs or reset env
    current_obs = start_obs.copy() if config.fixed_start_obs else env.reset()
    current_t = 0

    for i in range(config.num_iters):
        logging.info('---' * 5 + f' Start iteration i={i} ' + '---' * 5)
        logging.info(f'Length of data.x: {len(data.x)}')
        logging.info(f'Length of data.y: {len(data.y)}')

        # =====================================================
        #   Figure out what the next point to query should be
        # =====================================================
        # exe_path_list can be [] if there are no paths
        # model can be None if it isn't needed here
        x_next, exe_path_list, model = get_next_point(
                i,
                config,
                algo,
                domain,
                current_obs,
                env.action_space,
                gp_model_class,
                gp_model_params,
                acqfn_class,
                acqfn_params,
                acqopt_class,
                acqopt_params,
                deepcopy(data),
                dumper)

        # ==============================================
        #   Periodically run evaluation and plot
        # ==============================================
        if i % config.eval_frequency == 0 or i + 1 == config.num_iters:
            if model is None:
                model = gp_model_class(gp_model_params, data)
            # =======================================================================
            #    Evaluate MPC:
            #       - see how the MPC policy performs on the real env
            #       - see how well the model fits data from different distributions
            # =======================================================================
            real_paths_mpc = evaluate_mpc(
                    config,
                    test_algo,
                    model,
                    start_obs,
                    env,
                    f,
                    dumper,
                    data,
                    test_data,
                    test_mpc_data,
                    )

            # ============
            # Make Plots:
            #     - Posterior Mean paths
            #     - Posterior Sample paths
            #     - Observations
            #     - All of the above
            # ============
            make_plots(
                    plot_fn,
                    domain,
                    true_path,
                    data,
                    env,
                    config,
                    exe_path_list,
                    real_paths_mpc,
                    x_next,
                    dumper,
                    i,
                    )

        # Query function, update data
        y_next = f([x_next])[0]
        data.x.append(x_next)
        data.y.append(y_next)
        dumper.add('x', x_next)
        dumper.add('y', y_next)

        # for some setups we need to update the current state so that the policy / sampling
        # happens in the right place
        if config.alg.rollout_sampling:
            current_t += 1
            if current_t > env.horizon:
                current_t = 0
                current_obs = start_obs.copy() if config.fixed_start_obs else env.reset()
            else:
                delta = y_next[-obs_dim:]
                current_obs = update_fn(current_obs, delta)

        # Dumper save
        dumper.save()
        plt.close('all')


def configure(config):
    # Set plot settings
    neatplot.set_style()
    neatplot.update_rc('figure.dpi', 120)
    neatplot.update_rc('text.usetex', False)

    # Set random seed
    seed = config.seed
    np.random.seed(seed)
    tf.random.set_seed(seed)
    tf.config.run_functions_eagerly(config.tf_eager)

    # Check fixed_start_obs and num_samples_mc compatability
    assert (not config.fixed_start_obs) or config.num_samples_mc == 1, f"Need to have a fixed start obs ({config.fixed_start_obs}) or only 1 mc sample ({config.num_samples_mc})"  # NOQA


def get_env(config):
    env = gym.make(config.env.name)
    env.seed(config.seed)
    # set plot fn
    plot_fn = partial(plotters[config.env.name], env=env)

    if not config.alg.learn_reward:
        if config.alg.gd_opt:
            reward_function = envs.tf_reward_functions[config.env.name]
        else:
            reward_function = envs.reward_functions[config.env.name]
    else:
        reward_function = None
    if config.normalize_env:
        env = NormalizedEnv(env)
        if reward_function is not None:
            reward_function = make_normalized_reward_function(env, reward_function, config.alg.gd_opt)
        plot_fn = make_normalized_plot_fn(env, plot_fn)
    if config.alg.learn_reward:
        f = get_f_batch_mpc_reward(env, use_info_delta=config.teleport)
    else:
        f = get_f_batch_mpc(env, use_info_delta=config.teleport)
    update_fn = make_update_obs_fn(env, teleport=config.teleport, use_tf=config.alg.gd_opt)
    p0 = env.reset
    return env, f, plot_fn, reward_function, update_fn, p0


def get_initial_data(config, env, f, domain, dumper, plot_fn):
    data = Namespace()
    if config.sample_init_initially:
        data.x = [np.concatenate([env.reset(), env.action_space.sample()]) for _ in range(config.num_init_data)]
    else:
        data.x = unif_random_sample_domain(domain, config.num_init_data)
    data.y = f(data.x)
    dumper.extend('x', data.x)
    dumper.extend('y', data.y)

    # Plot initial data (TODO, refactor plotting)
    ax_obs_init, fig_obs_init = plot_fn(path=None, domain=domain)
    if ax_obs_init is not None:
        plot(ax_obs_init, data.x, 'o', color='k', ms=1)
        fig_obs_init.suptitle("Initial Observations")
        neatplot.save_figure(str(dumper.expdir / 'obs_init'), 'png', fig=fig_obs_init)
    return data


def get_model(config, env, obs_dim, action_dim):
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
    gp_model_params = {'n_dimy': obs_dim, 'gp_params': gp_params}
    if config.alg.kgrl or config.alg.pilco:
        gp_model_class = TFMultiGpfsGp
    else:
        gp_model_class = BatchMultiGpfsGp
    return gp_model_class, gp_model_params


def get_acq_fn(config, horizon, p0, reward_fn, update_fn, obs_dim, action_dim,
               gp_model_class, gp_model_params):
    if config.alg.uncertainty_sampling:
        acqfn_params = {}
        acqfn_class = UncertaintySamplingAcqFunction
    elif config.alg.kgrl or config.alg.pilco:
        acqfn_params = {
                'num_fs': config.alg.num_fs,
                'num_s0': config.alg.num_s0,
                'num_sprime_samps': config.alg.num_sprime_samps,
                'rollout_horizon': horizon,
                'p0': p0,
                'reward_fn': reward_fn,
                'update_fn': update_fn,
                'gp_model_class': gp_model_class,
                'gp_model_params': gp_model_params,
                'verbose': False,
                }
        acqfn_class = KGRLAcqFunction if config.alg.kgrl else PILCOAcqFunction
    else:
        acqfn_params = {'n_path': config.n_paths, 'crop': True}
        acqfn_class = MultiBaxAcqFunction
    return acqfn_class, acqfn_params


def get_acq_opt(config, obs_dim, action_dim, env, start_obs):
    if config.alg.gd_opt:
        acqopt_params = {
                "learning_rate": config.alg.learning_rate,
                "num_steps": config.alg.num_steps,
                "obs_dim": obs_dim,
                "action_dim": action_dim,
                "num_sprime_samps": config.alg.num_sprime_samps,
                "policy_test_period": config.alg.policy_test_period,
                "num_eval_trials": config.num_eval_trials,
                "policies": KGAcqOptimizer.get_policies(config.n_rand_acqopt,
                                                        config.alg.num_sprime_samps,
                                                        obs_dim,
                                                        action_dim,
                                                        [128, 128])
            }
        if config.alg.policy_test_period != 0:
            eval_fn = partial(evaluate_policy, env=env, start_obs=start_obs, autobatch=True)
            acqopt_params['eval_fn'] = eval_fn
        else:
            acqopt_params['eval_fn'] = None
        try:
            acqopt_params['hidden_layer_sizes'] = config.alg.hidden_layer_sizes
        except Exception:
            pass
        acqopt_class = KGAcqOptimizer
    else:
        acqopt_params = {}
        acqopt_class = AcqOptimizer
    return acqopt_class, acqopt_params


def fit_hypers(config, fit_data, plot_fn, domain, expdir):
    # Use test_mpc_data to fit hyperparameters
    assert len(fit_data.x) <= 3000, "fit_data larger than preset limit (can cause memory issues)"

    logging.info('\n'+'='*60+'\n Fitting Hyperparameters\n'+'='*60)
    logging.info(f'Number of observations in fit_data: {len(fit_data.x)}')

    # Plot hyper fitting data
    ax_obs_hyper_fit, fig_obs_hyper_fit = plot_fn(path=None, domain=domain)
    x_obs = [xi[0] for xi in fit_data.x]
    y_obs = [xi[1] for xi in fit_data.x]
    if ax_obs_hyper_fit:
        plot(ax_obs_hyper_fit, x_obs, y_obs, 'o', color='k', ms=1)
        neatplot.save_figure(str(expdir / 'mpc_obs_hyper_fit'), 'png', fig=fig_obs_hyper_fit)

    # Perform hyper fitting
    for idx in range(len(fit_data.y[0])):
        data_fit = Namespace(x=fit_data.x, y=[yi[idx] for yi in fit_data.y])
        gp_params = get_gpflow_hypers_from_data(data_fit, print_fit_hypers=True,
                                                opt_max_iter=config.env.gp.opt_max_iter)
        logging.info(f'gp_params for output {idx} = {gp_params}')


def execute_gt_mpc(config, algo_class, algo_params, f, dumper, domain, plot_fn):
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
    dumper.add('GT Returns', returns, log_mean_std=True)
    dumper.add('Path Lengths', path_lengths, log_mean_std=True)
    all_x = []
    for fp in full_paths:
        all_x += fp.x
    all_x = np.array(all_x)
    logging.info(f"all_x.shape = {all_x.shape}")
    logging.info(f"all_x.min(axis=0) = {all_x.min(axis=0)}")
    logging.info(f"all_x.max(axis=0) = {all_x.max(axis=0)}")
    logging.info(f"all_x.mean(axis=0) = {all_x.mean(axis=0)}")
    logging.info(f"all_x.var(axis=0) = {all_x.var(axis=0)}")
    # Save groundtruth paths plot
    if fig_gt:
        fig_gt.suptitle("Ground Truth Eval")
        neatplot.save_figure(str(dumper.expdir / 'gt'), 'png', fig=fig_gt)

    return true_path, test_mpc_data


def get_next_point(
        i,
        config,
        algo,
        domain,
        current_obs,
        action_space,
        gp_model_class,
        gp_model_params,
        acqfn_class,
        acqfn_params,
        acqopt_class,
        acqopt_params,
        data,
        dumper,
        ):
    exe_path_list = []
    model = None
    if config.alg.use_acquisition:
        model = gp_model_class(gp_model_params, data)
        # Set and optimize acquisition function
        acqfn_base = acqfn_class(params=acqfn_params, model=model, algorithm=algo)
        if config.num_samples_mc != 1:
            acqfn = MCAcqFunction(acqfn_base, {"num_samples_mc": config.num_samples_mc})
        else:
            acqfn = acqfn_base
        acqopt = acqopt_class(params=acqopt_params)
        acqopt.initialize(acqfn)
        if config.alg.rollout_sampling:
            x_test = [np.concatenate([current_obs, action_space.sample()]) for _ in range(config.n_rand_acqopt)]
        elif config.alg.eig and config.sample_exe:
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
            exe_path_list = acqfn.exe_path_list
            # Store returns of posterior samples
            posterior_returns = [compute_return(output[2], 1) for output in acqfn.output_list]
            dumper.add('Posterior Returns', posterior_returns, verbose=(i % config.eval_frequency == 0))
        else:
            x_test = unif_random_sample_domain(domain, n=config.n_rand_acqopt)
        x_next, acq_val = acqopt.optimize(x_test)
        dumper.add('Acquisition Function Value', acq_val)
        if config.alg.kgrl:
            dumper.add("Bayes Risks", acqopt.risk_vals, verbose=False)
            dumper.add("Policy Returns", acqopt.eval_vals, verbose=False)
            dumper.add("Policy Return ndata", acqopt.eval_steps, verbose=False)

    elif config.alg.use_mpc:
        model = gp_model_class(gp_model_params, data)
        algo.initialize()

        policy = partial(algo.execute_mpc, f=make_postmean_fn(model, use_tf=config.alg.gd_opt))
        action = policy(current_obs)
        x_next = np.concatenate([current_obs, action])
    else:
        x_next = unif_random_sample_domain(domain, 1)[0]
    return x_next, exe_path_list, model


def evaluate_mpc(
        config,
        algo,
        model,
        start_obs,
        env,
        f,
        dumper,
        data,
        test_data,
        test_mpc_data,
        ):
    with Timer("Evaluate the current MPC policy"):
        # execute the best we can
        # this is required to delete the current execution path
        algo.initialize()

        postmean_fn = make_postmean_fn(model, use_tf=config.alg.gd_opt)
        policy = partial(algo.execute_mpc, f=postmean_fn)
        real_returns = []
        mses = []
        all_x_mpc = []
        all_y_hat_mpc = []
        all_y_mpc = []
        real_paths_mpc = []
        pbar = trange(config.num_eval_trials)
        for j in pbar:
            real_obs, real_actions, real_rewards = evaluate_policy(policy, env, start_obs=start_obs,
                                                                   mpc_pass=True)
            real_return = compute_return(real_rewards, 1)
            real_returns.append(real_return)
            real_path_mpc = Namespace()
            real_path_mpc.x = real_obs
            plan_set_size = sum([len(path.x) for path in algo.old_exe_paths])
            mpc_sample_indices = random.sample(range(plan_set_size), config.test_set_size)
            x_mpc = []
            y_hat_mpc = []
            for path in algo.old_exe_paths:
                x_mpc.extend(path.x)
                y_hat_mpc.extend(path.y)
            x_mpc = [x_mpc[i] for i in mpc_sample_indices]
            y_hat_mpc = [y_hat_mpc[i] for i in mpc_sample_indices]
            all_x_mpc.extend(x_mpc)
            all_y_hat_mpc.extend(y_hat_mpc)
            y_mpc = f(x_mpc)
            all_y_mpc.extend(y_mpc)
            mses.append(mse(y_mpc, y_hat_mpc))
            stats = {"Mean Return": np.mean(real_returns), "Std Return:": np.std(real_returns),
                     "Model MSE": np.mean(mses)}

            pbar.set_postfix(stats)
            real_paths_mpc.append(real_path_mpc)
        real_returns = np.array(real_returns)
        algo.old_exe_paths = []
        dumper.add('Eval Returns', real_returns, log_mean_std=True)
        dumper.add('Eval ndata', len(data.x))
        current_mpc_mse = np.mean(mses)
        current_mpc_likelihood = model_likelihood(model, all_x_mpc, all_y_mpc)
        test_y_hat = postmean_fn(test_data.x)
        random_mse = mse(test_data.y, test_y_hat)
        random_likelihood = model_likelihood(model, test_data.x, test_data.y)
        gt_mpc_y_hat = postmean_fn(test_mpc_data.x)
        gt_mpc_mse = mse(test_mpc_data.y, gt_mpc_y_hat)
        gt_mpc_likelihood = model_likelihood(model, test_mpc_data.x, test_mpc_data.y)
        dumper.add('Model MSE (current MPC)', current_mpc_mse)
        dumper.add('Model MSE (random test set)', random_mse)
        dumper.add('Model MSE (GT MPC)', gt_mpc_mse)
        dumper.add('Model Likelihood (current MPC)', current_mpc_likelihood)
        dumper.add('Model Likelihood (random test set)', random_likelihood)
        dumper.add('Model Likelihood (GT MPC)', gt_mpc_likelihood)
        return real_paths_mpc


def make_plots(
        plot_fn,
        domain,
        true_path,
        data,
        env,
        config,
        exe_path_list,
        real_paths_mpc,
        x_next,
        dumper,
        i,
        ):
    # Initialize various axes and figures
    ax_all, fig_all = plot_fn(path=None, domain=domain)
    ax_postmean, fig_postmean = plot_fn(path=None, domain=domain)
    ax_samp, fig_samp = plot_fn(path=None, domain=domain)
    ax_obs, fig_obs = plot_fn(path=None, domain=domain)
    # Plot true path and posterior path samples
    ax_all, fig_all = plot_fn(true_path, ax_all, fig_all, domain, 'true')
    if ax_all is None:
        return
    # Plot observations
    x_obs = make_plot_obs(data.x, env, config.env.normalize_env)
    scatter(ax_all, x_obs, color='grey', s=10, alpha=0.3)
    plot(ax_obs, x_obs, 'o', color='k', ms=1)

    # Plot execution path posterior samples
    for path in exe_path_list:
        ax_all, fig_all = plot_fn(path, ax_all, fig_all, domain, 'samp')
        ax_samp, fig_samp = plot_fn(path, ax_samp, fig_samp, domain, 'samp')

    # plot posterior mean paths
    for path in real_paths_mpc:
        ax_all, fig_all = plot_fn(path, ax_all, fig_all, domain, 'postmean')
        ax_postmean, fig_postmean = plot_fn(path, ax_postmean, fig_postmean, domain, 'samp')

    # Plot x_next
    x = make_plot_obs(x_next, env, config.env.normalize_env)
    scatter(ax_all, x, facecolors='deeppink', edgecolors='k', s=120, zorder=100)
    plot(ax_obs, x, 'o', mfc='deeppink', mec='k', ms=12, zorder=100)

    try:
        # set titles if there is a single axes
        ax_all.set_title(f"All - Iteration {i}")
        ax_postmean.set_title(f"Posterior Mean Eval - Iteration {i}")
        ax_samp.set_title(f"Posterior Samples - Iteration {i}")
        ax_obs.set_title(f"Observations - Iteration {i}")
    except AttributeError:
        # set titles for figures if they are multi-axes
        fig_all.suptitle(f"All - Iteration {i}")
        fig_postmean.suptitle(f"Posterior Mean Eval - Iteration {i}")
        fig_samp.suptitle(f"Posterior Samples - Iteration {i}")
        fig_obs.suptitle(f"Observations - Iteration {i}")

    # Save figure at end of evaluation
    neatplot.save_figure(str(dumper.expdir / f'all_{i}'), 'png', fig=fig_all)
    neatplot.save_figure(str(dumper.expdir / f'postmean_{i}'), 'png', fig=fig_postmean)
    neatplot.save_figure(str(dumper.expdir / f'samp_{i}'), 'png', fig=fig_samp)
    neatplot.save_figure(str(dumper.expdir / f'obs_{i}'), 'png', fig=fig_obs)


if __name__ == '__main__':
    main()
