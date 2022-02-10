'''
Weird gain env: an environment that allows for open- and closed-loop controllers
to be tested in an environment that is easy to plot and understand.

Viraj Mehta, 2022
'''

import logging
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np



class WeirdGainEnv(gym.Env):

    def __init__(self):
        self.observation_space = spaces.Box(low=np.array([-10, -10]), high=np.array([10, 10]))
        self.action_space = spaces.Box(low=-np.ones(2), high=np.ones(2))
        self.x = None
        self.start_space_low = np.array([-10, -10])
        self.start_space_high = np.array([-5, -5])

    def reset(self, obs=None):
        if obs is not None:
            self.x = np.random.uniform(self.start_space_low)
        else:
            self.x = obs

    def get_B(self):
        # TODO: make this more interesting
        return np.eye(2)

    def step(self, action):
        B = self.get_B()
        delta_x = B @ action
        self.x += delta_x
        rew = weird_gain_rew(self.x)
        return self.x, rew, False, {}
