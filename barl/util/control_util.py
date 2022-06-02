import numpy as np
from .misc_util import batch_function
import gym
import logging

try:
    # from gym.envs.mujoco.mujoco_env import MujocoEnv
    mj_here = False
except:
    mj_here = False
import colorednoise
from tqdm import tqdm, trange
from abc import ABC, abstractmethod


def choose_subset(data_list, idx):
    out = []
    for ix in idx:
        out.append(data_list[ix])
    return out


def get_f_mpc(env, use_info_delta=False):
    obs_dim = len(env.observation_space.low)

    def f(x):
        x = np.array(x)
        obs = x[:obs_dim]
        action = x[obs_dim:]
        env.reset(obs)
        next_obs, reward, done, info = env.step(action)
        if use_info_delta:
            return info["delta_obs"]
        else:
            return next_obs - obs

    return f


def get_f_batch_mpc(env, **kwargs):
    return batch_function(get_f_mpc(env, **kwargs))


def get_f_mpc_reward(env, use_info_delta=False):
    obs_dim = len(env.observation_space.low)

    def f(x):
        x = np.array(x)
        obs = x[:obs_dim]
        action = x[obs_dim:]
        env.reset(obs)
        next_obs, reward, done, info = env.step(action)
        if use_info_delta:
            delta_obs = info["delta_obs"]
        else:
            delta_obs = next_obs - obs
        return np.insert(delta_obs, 0, reward)

    return f


def get_f_batch_mpc_reward(env, **kwargs):
    return batch_function(get_f_mpc_reward(env, **kwargs))


def CEM(
    start_obs,
    action_dim,
    dynamics_unroller,
    horizon,
    alpha,
    popsize,
    elite_frac,
    num_iters,
    verbose=False,
):
    """
    CEM: the cross-entropy method, here used for planning optimal actions on an MDP.
    assumes action space is [-1, 1]^action_dim
    """
    action_upper_bound = 1
    action_lower_bound = -1
    initial_variance_divisor = 4
    num_elites = int(popsize * elite_frac)
    mean = np.zeros(action_dim)
    var = (
        np.ones_like(mean)
        * ((action_upper_bound - action_lower_bound) / initial_variance_divisor) ** 2
    )
    best_sample, best_obs, best_return = None, None, -np.inf
    for i in trange(num_iters, disable=not verbose):
        samples = (
            np.fmod(np.random.normal(size=(popsize, horizon, action_dim)), 2)
            * np.sqrt(var)
            + mean
        )
        samples = np.clip(samples, action_lower_bound, action_upper_bound)
        observations, returns = dynamics_unroller(start_obs, samples)
        elites = samples[np.argsort(returns)[-num_elites:], ...]
        new_mean = np.mean(elites, axis=0)
        new_var = np.var(elites, axis=0)
        mean = alpha * mean + (1 - alpha) * new_mean
        var = alpha * var + (1 - alpha) * new_var
        best_idx = np.argmax(returns)
        best_current_return = returns[best_idx]
        if best_current_return > best_return:
            best_return = best_current_return
            best_sample = samples[best_idx, ...]
            best_obs = observations[best_idx]
    return best_return, best_obs, best_sample


def iCEM_generate_samples(
    nsamps, horizon, beta, mean, var, action_lower_bound, action_upper_bound
):
    action_dim = mean.shape[-1]
    samples = (
        colorednoise.powerlaw_psd_gaussian(
            beta, size=(nsamps, action_dim, horizon)
        ).transpose([0, 2, 1])
        * np.sqrt(var)
        + mean
    )
    samples = np.clip(samples, action_lower_bound, action_upper_bound)
    return samples


def iCEM(
    start_obs,
    action_dim,
    dynamics_unroller,
    num_samples,
    horizon,
    n_elites,
    beta,
    num_iters,
    gamma,
    mem_fraction=0.3,
    prev_samples=None,
    verbose=False,
):
    action_upper_bound = 1
    action_lower_bound = -1
    initial_variance_divisor = 4
    mean = np.zeros(action_dim)
    var = (
        np.ones_like(mean)
        * ((action_upper_bound - action_lower_bound) / initial_variance_divisor) ** 2
    )
    elites, elite_observations, elite_returns = None, None, None
    best_sample, best_obs, best_return = None, None, -np.inf
    for i in trange(num_iters, disable=not verbose):
        num_traj = int(max(num_samples * (gamma**-i), 2 * n_elites))
        samples = iCEM_generate_samples(
            num_traj, horizon, beta, mean, var, action_lower_bound, action_upper_bound
        )
        if i == 0 and prev_samples is not None:
            bs = prev_samples.shape[0]
            shifted_samples = np.concatenate(
                [prev_samples[:, 1:, :], np.zeros((bs, 1, action_dim))], axis=1
            )
            shifted_subset = shifted_samples[
                np.random.choice(bs, int(bs * mem_fraction), replace=False), ...
            ]
            samples = np.concatenate([samples, shifted_subset], axis=0)
        if i + 1 == num_iters:
            samples = np.concatenate([samples, mean[None, :]], axis=0)
        observations, returns = dynamics_unroller(start_obs, samples)
        if i > 0:
            elite_subset_idx = np.random.choice(
                n_elites, int(n_elites * mem_fraction), replace=False
            )
            elite_subset = elites[elite_subset_idx, ...]
            elite_obs_subset = choose_subset(elite_observations, elite_subset_idx)
            elite_return_subset = elite_returns[elite_subset_idx]
            samples = np.concatenate([samples, elite_subset], axis=0)
            observations = observations + elite_obs_subset
            returns = np.concatenate([returns, elite_return_subset])
        elite_idx = np.argsort(returns)[-n_elites:]
        elites = samples[elite_idx, ...]
        elite_observations = choose_subset(observations, elite_idx)
        elite_returns = returns[elite_idx]
        mean = np.mean(elites, axis=0)
        var = np.var(elites, axis=0)
        best_idx = np.argmax(returns)
        best_current_return = returns[best_idx]
        if best_current_return > best_return:
            best_return = best_current_return
            best_sample = samples[best_idx, ...]
            best_obs = observations[best_idx]
    return best_return, best_obs, best_sample, elites


def compute_return(rewards, discount_factor):
    rewards = np.array(rewards)
    if rewards.ndim == 2:
        rewards = rewards.T
    return np.polynomial.polynomial.polyval(discount_factor, rewards)


class DynamicsUnroller(ABC):
    def __init__(self, gamma):
        self.gamma = gamma

    def __call__(self, start_obs, action_samples):
        all_observations, all_returns = [], []
        for sample in action_samples:
            observations, rewards = self.unroll(start_obs, sample)
            all_observations.append(observations)
            all_returns.append(self.compute_return(rewards))
        return all_observations, np.array(all_returns)

    @abstractmethod
    def unroll(self, start_obs, action_samples):
        pass

    def compute_return(self, rewards):
        return np.polynomial.polynomial.polyval(self.gamma, rewards)


class EnvDynamicsUnroller(DynamicsUnroller):
    def __init__(self, env, gamma=0.99, verbose=False):
        super().__init__(gamma)
        self._env = env
        self.silent = not verbose
        self.query_count = 0

    def unroll(self, start_obs, action_samples):
        observations = [self._env.reset(start_obs)]
        rewards = []
        for action in tqdm(action_samples, disable=self.silent):
            self.query_count += 1
            obs, rew, done, info = self._env.step(action)
            observations.append(obs)
            rewards.append(rew)
            if done:
                break
        return observations, rewards


class ResettableEnv(gym.Env):
    def __init__(self, env):
        self._wrapped_env = env
        self.action_space = self._wrapped_env.action_space
        self.observation_space = self._wrapped_env.observation_space
        if mj_here:
            self.is_mujoco = isinstance(env, MujocoEnv)
        else:
            self.is_mujoco = False
        self.npos = len(env.init_qpos) if self.is_mujoco else None

    @property
    def wrapped_env(self):
        return self._wrapped_env

    def reset(self, obs=None, **kwargs):
        reset_obs = self._wrapped_env.reset(**kwargs)
        if obs is not None and not self.is_mujoco:
            obs = np.array(obs)
            self._wrapped_env.state = obs
            return obs
        elif obs is not None:
            obs = np.array(obs)
            qpos = obs[: self.npos]
            qvel = obs[self.npos :]
            self._wrapped_env.set_state(qpos, qvel)
            return obs
        return reset_obs

    def step(self, action):
        return self._wrapped_env.step(action)

    def render(self, *args, **kwargs):
        return self._wrapped_env.render(*args, **kwargs)

    @property
    def horizon(self):
        return self._wrapped_env.horizon

    def __getattr__(self, attr):
        if attr == "_wrapped_env":
            raise AttributeError()
        return getattr(self._wrapped_env, attr)


def rollout_cem_continuous_cartpole(env, unroller):
    start_obs = env.reset()
    action_dim = 1
    horizon = 10
    alpha = 0.2
    popsize = 50
    elite_frac = 0.1
    n_iters = 5
    done = False
    rewards = []
    env_horizon = env.horizon
    for _ in trange(env_horizon):
        seq_return, observations, actions = CEM(
            start_obs,
            action_dim,
            unroller,
            horizon,
            alpha,
            popsize,
            elite_frac,
            n_iters,
        )
        action = actions[0]
        start_obs, rew, done, info = env.step(action)
        rewards.append(rew)
        if done:
            break
    return sum(rewards)


def rollout_icem_continuous_cartpole(env, unroller):
    start_obs = env.reset()
    action_dim = 1
    budget = 8
    horizon = 10
    n_elites = 4
    beta = 3
    gamma = 1.25
    num_iters = 3
    actions_per_plan = 4
    done = False
    rewards = []
    env_horizon = env.horizon
    elites = None
    for _ in trange(env_horizon):
        seq_return, observations, actions, elites = iCEM(
            start_obs,
            action_dim,
            unroller,
            budget,
            horizon,
            n_elites,
            beta,
            num_iters,
            gamma,
            prev_samples=elites,
        )
        for i in range(actions_per_plan):
            action = actions[i]
            start_obs, rew, done, info = env.step(action)
            rewards.append(rew)
            if done:
                return sum(rewards)
    return sum(rewards)


def evaluate_policy(policy, env, start_obs=None, mpc_pass=False, autobatch=False):
    obs = env.reset(start_obs)
    observations = [obs]
    actions = []
    rewards = []
    done = False
    samples_to_pass = []
    for _ in range(env.horizon):
        if autobatch:
            obs = obs[None, :]
        if not mpc_pass:
            action = policy(obs)
        else:
            action, samples_to_pass = policy(
                obs, samples_to_pass=samples_to_pass, return_samps=True
            )
        if autobatch:
            action = action[0, :]
        obs, rew, done, info = env.step(action)
        observations.append(obs)
        actions.append(action)
        rewards.append(float(rew))
        if done:
            break
    return observations, actions, rewards


def test_continuous_cartpole():
    from envs.continuous_cartpole import ContinuousCartPoleEnv

    # from pets_cartpole import PETSCartpoleEnv
    algo = "iCEM"
    fn = (
        rollout_cem_continuous_cartpole
        if algo == "CEM"
        else rollout_icem_continuous_cartpole
    )
    env = ContinuousCartPoleEnv()
    plan_env = ResettableEnv(ContinuousCartPoleEnv())
    unroller = EnvDynamicsUnroller(plan_env)
    query_counts = []
    returns = []
    neps = 5
    pbar = trange(neps)
    for _ in pbar:
        unroller.query_count = 0
        rollout_return = fn(env, unroller)
        returns.append(rollout_return)
        query_counts.append(unroller.query_count)
        pbar.set_postfix(
            ordered_dict={
                "Mean Return": np.mean(returns),
                "Mean Query Count": np.mean(query_counts),
            }
        )

    returns = np.array(returns)
    query_counts = np.array(query_counts)
    logging.info(
        f"{algo} gets {returns.mean():.1f} mean return with stderr {returns.std() / np.sqrt(neps):.1f}"
    )
    logging.info(
        f"{algo} uses {query_counts.mean():.1f} queries per trial with stderr {query_counts.std() / np.sqrt(neps):.1f}"
    )  # NOQA


if __name__ == "__main__":
    test_continuous_cartpole()
