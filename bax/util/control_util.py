import numpy as np
import gym
from tqdm import tqdm, trange
from abc import ABC, abstractmethod


def CEM(start_obs,
        action_dim,
        dynamics_unroller,
        horizon,
        alpha,
        popsize,
        elite_frac,
        num_iters,
        verbose=False):
    '''
    CEM: the cross-entropy method, here used for planning optimal actions on an MDP.
    assumes action space is [-1, 1]^action_dim
    '''
    action_upper_bound = 1
    action_lower_bound = -1
    initial_variance_divisor = 4
    num_elites = int(popsize * elite_frac)
    mean = np.zeros(action_dim)
    var = np.ones_like(mean) * ((action_upper_bound - action_lower_bound) / initial_variance_divisor) ** 2
    best_sample, best_obs, best_return = None, None, -np.inf
    for i in trange(num_iters, disable=not verbose):
        samples = np.fmod(np.random.normal(size=(popsize, horizon, action_dim)), 2) * np.sqrt(var) + mean
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


class DynamicsUnroller(ABC):
    def __init__(self, gamma):
        self.gamma = gamma

    def __call__(self, start_obs, action_samples):
        all_observations, all_returns = [], []
        for sample in action_samples:
            observations, rewards = self.unroll(start_obs, sample)
            all_observations.append(observations)
            all_returns.append(self.compute_return(rewards))
        return all_observations, all_returns

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

    @property
    def wrapped_env(self):
        return self._wrapped_env

    def reset(self, obs=None, **kwargs):
        reset_obs = self._wrapped_env.reset(**kwargs)
        if obs is not None:
            obs = np.array(obs)
            self._wrapped_env.state = obs
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
        if attr == '_wrapped_env':
            raise AttributeError()
        return getattr(self._wrapped_env, attr)


def rollout_cem_continuous_cartpole(env, unroller):
    start_obs = env.reset()
    action_dim = 1
    horizon = 10
    alpha = 0.25
    popsize = 100
    elite_frac = 0.25
    n_iters = 5
    done = False
    rewards = []
    env_horizon = env.horizon
    for _ in trange(env_horizon):
        seq_return, observations, actions = CEM(start_obs, action_dim, unroller, horizon, alpha,
                                                popsize, elite_frac, n_iters)
        action = actions[0]
        start_obs, rew, done, info = env.step(action)
        rewards.append(rew)
        if done:
            break
    return sum(rewards)


def test_cem_continuous_cartpole():
    from continuous_cartpole import ContinuousCartPoleEnv
    env = ContinuousCartPoleEnv()
    plan_env = ContinuousCartPoleEnv()
    unroller = EnvDynamicsUnroller(plan_env)
    query_counts = []
    returns = []
    neps = 25
    for _ in trange(neps):
        unroller.query_count = 0
        rollout_return = rollout_cem_continuous_cartpole(env, unroller)
        returns.append(rollout_return)
        query_counts.append(unroller.query_count)
    returns = np.array(returns)
    query_counts = np.array(query_counts)
    print(f"CEM gets {returns.mean():.1f} mean return with std {returns.std():.1f}")
    print(f"CEM uses {query_counts.mean():.1f} queries per trial with std {query_counts.std():.1f}")


if __name__ == '__main__':
    test_cem_continuous_cartpole()
