import gym
from gym.envs.registration import register
from gym import Env, spaces
import numpy as np


class TrigWrapperEnv(Env):
    def __init__(self, base_name):
        self._wrapped_env = gym.make(base_name)
        self.action_space = self._wrapped_env.action_space
        wrapped_obs_space = self._wrapped_env.observation_space
        high = []
        low = []
        for i in range(wrapped_obs_space.high.size):
            if i in self._wrapped_env.periodic_dimensions:
                high += [1, 1]
                low += [-1, -1]
            else:
                high.append(wrapped_obs_space.high[i])
                low.append(wrapped_obs_space.low[i])

        self.prev_obs = None
        self.observation_space = spaces.Box(high=np.array(high), low=np.array(low))
        self.wrapped_periodic_dimensions = self._wrapped_env.periodic_dimensions
        self.periodic_dimensions = []

    def reset(self, obs=None):
        if obs is None:
            trig_obs = angle_to_trig(self._wrapped_env.reset(), trig_dims=self.wrapped_periodic_dimensions)
            self.prev_obs = trig_obs.copy()
            return trig_obs
        norm_obs = trig_to_angle(obs, trig_dims=self.wrapped_periodic_dimensions)
        self.prev_obs = obs.copy()
        return angle_to_trig(self._wrapped_env.reset(norm_obs), trig_dims=self.wrapped_periodic_dimensions)

    def step(self, action):
        obs, rew, done, info = self._wrapped_env.step(action)
        info['angle_obs'] = obs.copy()
        # need to handle delta_obs
        norm_obs = angle_to_trig(obs, trig_dims=self.wrapped_periodic_dimensions)
        if 'delta_obs' in info:
            info['delta_obs'] = norm_obs - self.prev_obs
        self.prev_obs = norm_obs
        return norm_obs, rew, done, info

    @property
    def horizon(self):
        return self._wrapped_env.horizon

    def terminate(self):
        if hasattr(self.wrapped_env, "terminate"):
            self.wrapped_env.terminate()


def get_theta(s, c):
    r_sq = s ** 2 + c ** 2
    r = np.sqrt(r_sq)
    abs_theta = np.arccos(c / r)
    return abs_theta * np.sign(s)


def trig_to_angle(obs, trig_dims):
    obs = obs.copy()
    for dim in trig_dims:
        s = obs[..., dim]
        c = obs[..., dim + 1]
        theta = get_theta(s, c)
        obs[..., dim] = theta
        obs = np.delete(obs, dim + 1, axis=-1)
    return obs


def angle_to_trig(obs, trig_dims):
    obs = obs.copy()
    for dim in reversed(trig_dims):
        theta = obs[..., dim]
        s = np.sin(theta)
        c = np.cos(theta)
        obs[..., dim] = s
        obs = np.insert(obs, dim + 1, c, axis=-1)
    return obs


def make_trig_reward_function(periodic_dimensions, reward_function):

    def trig_rew_fn(x, y):
        angle_x = trig_to_angle(x, periodic_dimensions)
        angle_y = trig_to_angle(y, periodic_dimensions)
        return reward_function(angle_x, angle_y)
    return trig_rew_fn


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


def test_normalization():
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
    for _ in range(wrapped_env.horizon):
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
        if done:
            break
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


def test_trig_wrapper():
    from pendulum import PendulumEnv, pendulum_reward
    register(
        id='bacpendulum-v0',
        entry_point=PendulumEnv,
        )
    env = TrigWrapperEnv('bacpendulum-v0')
    old_obs = env.reset()
    action = env.action_space.sample()
    obs, rew, done, info = env.step(action)
    angle_to_trig_obs = angle_to_trig(info['angle_obs'], env.periodic_dimensions)
    assert np.allclose(angle_to_trig_obs, obs), f"angle_to_trig obs: {angle_to_trig_obs}, obs: {obs}"
    trig_to_angle_obs = trig_to_angle(obs, env.periodic_dimensions)
    assert np.allclose(trig_to_angle_obs, info['angle_obs']), f"obs: {obs}, trig_to_angle_obs: {trig_to_angle_obs}, angle_obs: {info['angle_obs']}"
    env.reset()
    new_obs = env.reset(old_obs)
    assert np.allclose(new_obs, old_obs), f"new_obs: {new_obs}, old_obs: {old_obs}"
    half_trig_obs = old_obs.copy()
    half_trig_obs[0:2] /= 2
    scaled_obs = env.reset(half_trig_obs)
    assert np.allclose(old_obs, scaled_obs), f"half_trig_obs: {half_trig_obs}, old_obs: {old_obs}, scaled_obs: {scaled_obs}"
    trig_rew_fn = make_trig_reward_function(env.periodic_dimensions, pendulum_reward)
    xs = []
    ys = []
    rewards = []
    for _ in range(env.horizon):
        action = env.action_space.sample()
        obs, rew, done, info = env.step(action)
        angle_to_trig_obs = angle_to_trig(info['angle_obs'], env.periodic_dimensions)
        assert np.allclose(angle_to_trig_obs, obs), f"angle_to_trig obs: {angle_to_trig_obs}, obs: {obs}"
        trig_to_angle_obs = trig_to_angle(obs, env.periodic_dimensions)
        assert np.allclose(trig_to_angle_obs, info['angle_obs']), f"obs: {obs}, trig_to_angle_obs: {trig_to_angle_obs}, angle_obs: {info['angle_obs']}"
        x = np.concatenate([old_obs, action])
        xs.append(x)
        ys.append(obs)
        rew_hat = trig_rew_fn(x, obs)
        rewards.append(rew)
        assert np.allclose(rew_hat, rew)
        old_obs = obs

    xs = np.vstack(xs)
    ys = np.vstack(ys)
    rewards = np.array(rewards)
    rewards_hat = trig_rew_fn(xs, ys)
    assert np.allclose(rewards, rewards_hat)



if __name__ == "__main__":
    # test_normalization()
    test_trig_wrapper()
