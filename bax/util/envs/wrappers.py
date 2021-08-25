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
        if 'delta_obs' in info:
            unnorm_delta_obs = info['delta_obs']
            norm_delta_obs = unnorm_delta_obs / self.unnorm_obs_space_size * 2
            info['delta_obs'] = norm_delta_obs
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
        rewards = reward_function(unnorm_x, unnorm_y)
        return rewards
    return norm_rew_fn


def make_update_obs_fn(env, teleport=False):
    periods = []
    obs_dim = env.observation_space.low.size
    obs_range = env.observation_space.high - env.observation_space.low
    for i in range(obs_dim):
        if i in env.periodic_dimensions:
            periods.append(env.observation_space.high[i] - env.observation_space.low[i])
        else:
            periods.append(0)
    periods = np.array(periods)
    periodic = periods != 0
    def update_obs_fn(x, y):
        start_obs = x[..., :obs_dim]
        delta_obs = y[..., -obs_dim:]
        output = start_obs + delta_obs
        if not teleport:
            return output
        shifted_output = output - env.observation_space.low
        if x.ndim == 2:
            mask = np.tile(periodic, (x.shape[0], 1))
        else:
            mask = periodic
        np.remainder(shifted_output, obs_range, where=mask, out=shifted_output)
        modded_output = shifted_output
        wrapped_output = modded_output + env.observation_space.low
        return wrapped_output
    return update_obs_fn


def test_obs(wrapped_env, obs):
    unnorm_obs = wrapped_env.unnormalize_obs(obs)
    renorm_obs = wrapped_env.normalize_obs(unnorm_obs)
    assert np.allclose(obs, renorm_obs), f"Original obs {obs} not close to renormalized obs {renorm_obs}"


def test_rew_fn(gt_rew, norm_rew_fn, old_obs, action, obs):
    x = np.concatenate([old_obs, action])
    y = obs
    assert np.allclose(gt_rew, norm_rew_fn(x, y))


def test_update_function(start_obs, action, delta_obs, next_obs, update_fn):
    x = np.concatenate([start_obs, action], axis=-1)
    updated_next_obs = update_fn(x, delta_obs)
    assert np.allclose(next_obs, updated_next_obs), f"Next obs: {next_obs} and updated next obs: {updated_next_obs}"


def test():
    import sys
    sys.path.append('.')
    from pendulum import PendulumEnv, pendulum_reward
    sys.path.append('..')
    env = PendulumEnv()
    wrapped_env = NormalizedEnv(env)
    regular_update_fn = make_update_obs_fn(wrapped_env)
    wrapped_reward = make_normalized_reward_function(wrapped_env, pendulum_reward)
    teleport_update_fn = make_update_obs_fn(wrapped_env, teleport=True)
    obs = wrapped_env.reset()
    test_obs(wrapped_env, obs)
    done = False
    total_rew = 0
    observations = []
    next_observations = []
    rewards = []
    actions = []
    teleport_deltas = []
    while not done:
        old_obs = obs
        observations.append(old_obs)
        action = wrapped_env.action_space.sample()
        actions.append(action)
        obs, rew, done, info = wrapped_env.step(action)
        next_observations.append(obs)
        total_rew += rew
        standard_delta_obs = obs - old_obs
        teleport_deltas.append(info['delta_obs'])
        test_update_function(old_obs, action, standard_delta_obs, obs, regular_update_fn)
        test_update_function(old_obs, action, info['delta_obs'], obs, teleport_update_fn)
        rewards.append(rew)
        test_obs(wrapped_env, obs)
        test_rew_fn(rew, wrapped_reward, old_obs, action, obs)
    observations = np.array(observations)
    actions = np.array(actions)
    rewards = np.array(rewards)
    next_observations = np.array(next_observations)
    teleport_deltas = np.array(teleport_deltas)
    x = np.concatenate([observations, actions], axis=1)
    teleport_next_obs = teleport_update_fn(x, teleport_deltas)
    assert np.allclose(teleport_next_obs, next_observations)
    test_rewards = wrapped_reward(x, next_observations)
    assert np.allclose(rewards, test_rewards), f"Rewards: {rewards} not equal to test rewards: {test_rewards}"
    print(f"passed!, rew={total_rew}")

if __name__ == "__main__":
    test()
