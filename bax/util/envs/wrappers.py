import gym
from gym import Env, spaces
import numpy as np


class NormalizedEnv(Env):
    def __init__(self, wrapped_env):
        '''
        Normalizes obs to be between -1 and 1
        doesn't touch actions
        '''
        self._wrapped_env = wrapped_env
        self.unnorm_action_space = self._wrapped_env.action_space
        self.unnorm_observation_space = self._wrapped_env.observation_space
        self.unnorm_obs_space_size = self.unnorm_observation_space.high - self.unnorm_observation_space.low
        self.unnorm_action_space_size = self.unnorm_action_space.high - self.unnorm_action_space.low
        self.action_space = spaces.Box(low=-np.ones_like(self.unnorm_action_space.low),
                                            high=np.ones_like(self.unnorm_action_space.high))
        self.observation_space = spaces.Box(low=-np.ones_like(self.unnorm_observation_space.low),
                                            high=np.ones_like(self.unnorm_observation_space.high))

    @property
    def wrapped_env(self):
        return self._wrapped_env

    def reset(self, obs=None):
        if obs is not None:
            unnorm_obs = self.unnormalize_obs(obs)
            unnorm_obs = self._wrapped_env.reset(obs=unnorm_obs)
        else:
            unnorm_obs = self._wrapped_env.reset()
        return self.normalize_obs(unnorm_obs)

    def step(self, action):
        unnorm_action = self.unnormalize_action(action)
        unnorm_obs, rew, done, info = self._wrapped_env.step(unnorm_action)
        return self.normalize_obs(unnorm_obs), rew, done, info

    def render(self, *args, **kwargs):
        return self._wrapped_env.render(*args, **kwargs)

    @property
    def horizon(self):
        return self._wrapped_env.horizon

    def terminate(self):
        if hasattr(self.wrapped_env, "terminate"):
            self.wrapped_env.terminate()

    def __getattr__(self, attr):
        if attr == '_wrapped_env':
            raise AttributeError()
        return getattr(self._wrapped_env, attr)

    def __getstate__(self):
        """
        This is useful to override in case the wrapped env has some funky
        __getstate__ that doesn't play well with overriding __getattr__.

        The main problematic case is/was gym's EzPickle serialization scheme.
        :return:
        """
        return self.__dict__

    def __setstate__(self, state):
        self.__dict__.update(state)

    def __str__(self):
        return '{}({})'.format(type(self).__name__, self.wrapped_env)

    def normalize_obs(self, obs):
        if obs.ndim == 1:
            low = self.unnorm_observation_space.low
            size = self.unnorm_obs_space_size
        else:
            low = self.unnorm_observation_space.low[None, :]
            size = self.unnorm_obs_space_size[None, :]
        pos_obs = obs - low
        norm_obs = (pos_obs / size * 2) - 1
        return norm_obs

    def unnormalize_obs(self, obs):
        if obs.ndim == 1:
            low = self.unnorm_observation_space.low
            size = self.unnorm_obs_space_size
        else:
            low = self.unnorm_observation_space.low[None, :]
            size = self.unnorm_obs_space_size[None, :]
        obs01 = (obs + 1) / 2
        obs_ranged = obs01 * size
        unnorm_obs = obs_ranged + low
        return unnorm_obs

    def unnormalize_action(self, action):
        if action.ndim == 1:
            low = self.unnorm_action_space.low
            size = self.unnorm_action_space_size
        else:
            low = self.unnorm_action_space.low[None, :]
            size = self.unnorm_action_space_size[None, :]
        act01 = (action + 1) / 2
        act_ranged = act01 * size
        unnorm_act = act_ranged + low
        return unnorm_act

def make_normalized_reward_function(norm_env, reward_function):
    '''
    reward functions always take x, y as args
    x: [obs; action]
    y: [next_obs]
    this assumes obs and next_obs are normalized but the reward function handles them in unnormalized form
    '''
    obs_dim = norm_env.observation_space.low.size
    def norm_rew_fn(x, y):
        norm_obs = x[..., :obs_dim]
        action = x[..., obs_dim:]
        unnorm_action = norm_env.unnormalize_action(action)
        unnorm_obs = norm_env.unnormalize_obs(norm_obs)
        unnorm_x = np.concatenate([unnorm_obs, unnorm_action], axis=-1)
        unnorm_y = norm_env.unnormalize_obs(y)
        return reward_function(unnorm_x, unnorm_y)
    return norm_rew_fn

def test_obs(wrapped_env, obs):
    unnorm_obs = wrapped_env.unnormalize_obs(obs)
    renorm_obs = wrapped_env.normalize_obs(unnorm_obs)
    assert np.allclose(obs, renorm_obs), f"Original obs {obs} not close to renormalized obs {renorm_obs}"

def test_rew_fn(gt_rew, norm_rew_fn, old_obs, action, obs):
    x = np.concatenate([old_obs, action])
    y = obs
    assert np.allclose(gt_rew, norm_rew_fn(x, y))

def test():
    import sys
    sys.path.append('.')
    from pendulum import PendulumEnv, pendulum_reward
    env = PendulumEnv()
    wrapped_env = NormalizedEnv(env)
    wrapped_reward = make_normalized_reward_function(wrapped_env, pendulum_reward)
    obs = wrapped_env.reset()
    test_obs(wrapped_env, obs)
    done = False
    while not done:
        old_obs = obs
        action = wrapped_env.action_space.sample()
        obs, rew, done, info = wrapped_env.step(action)
        test_obs(wrapped_env, obs)
        test_rew_fn(rew, wrapped_reward, old_obs, action, obs)
    print("passed!")

if __name__ == "__main__":
    test()
