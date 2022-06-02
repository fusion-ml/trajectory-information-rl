"""
Weird gain env: an environment that allows for open- and closed-loop controllers
to be tested in an environment that is easy to plot and understand.

Viraj Mehta, 2022
"""

import gym
from gym import spaces
import numpy as np

GOAL = np.array([6, 9])


class WeirdGainEnv(gym.Env):
    def __init__(self):
        self.observation_space = spaces.Box(
            low=np.array([-10, -10]), high=np.array([10, 10])
        )
        self.action_space = spaces.Box(low=-np.ones(2), high=np.ones(2))
        self.x = None
        self.start_space_low = np.array([-10, -6])
        self.start_space_high = np.array([-5, -5])
        self.periodic_dimensions = []
        self.horizon = 10

    def reset(self, obs=None):
        if obs is None:
            self.x = np.random.uniform(self.start_space_low)
        else:
            self.x = obs
        return self.x

    def get_B(self):
        # just some arbitrary continuous function from state to 2x2 mx
        x_gain = np.sin(self.x[1] * np.pi / 10) * 2
        y_gain = np.cos(self.x[0] * np.pi / 10) * 2
        scaling = np.array([[x_gain, 0], [0, y_gain]])

        return scaling

    def step(self, action):
        B = self.get_B()
        delta_x = B @ action
        self.x += delta_x
        self.x = np.clip(
            self.x, self.observation_space.low, self.observation_space.high
        )
        rew = _weird_gain_rew(self.x)
        return self.x, rew, False, {}


class WeirderGainEnv(gym.Env):
    def __init__(self):
        self.observation_space = spaces.Box(
            low=np.array([-10, -10]), high=np.array([10, 10])
        )
        self.action_space = spaces.Box(low=-np.ones(2), high=np.ones(2))
        self.x = None
        self.start_space_low = np.array([-10, -6])
        self.start_space_high = np.array([-5, -5])
        self.periodic_dimensions = []
        self.horizon = 10

    def reset(self, obs=None):
        if obs is None:
            self.x = np.random.uniform(self.start_space_low)
        else:
            self.x = obs
        return self.x

    def get_B(self):
        # just some arbitrary continuous function from state to 2x2 mx
        x_gain = np.sin(self.x[1] * np.pi / 5) * 3 + np.cos(self.x[1] * np.pi / 3)
        y_gain = np.cos(self.x[0] * np.pi / 7) * 3
        scaling = np.array([[x_gain, 0], [0, y_gain]])
        return scaling

    def step(self, action):
        B = self.get_B()
        delta_x = B @ action
        self.x += delta_x
        self.x = np.clip(
            self.x, self.observation_space.low, self.observation_space.high
        )
        rew = _weird_gain_rew(self.x)
        return self.x, rew, False, {}


def _weird_gain_rew(x):
    return -np.sum(np.abs(x - GOAL), axis=-1)


def weird_gain_reward(x, next_obs):
    return _weird_gain_rew(next_obs)
