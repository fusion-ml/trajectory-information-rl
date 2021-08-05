from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os

import numpy as np
from gym import utils, spaces
from gym.envs.mujoco import mujoco_env


class PETSCartpoleEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    PENDULUM_LENGTH = 0.6
    OBSERVATION_DIM = 4
    ACTION_DIM = 1

    def __init__(self):
        utils.EzPickle.__init__(self)
        dir_path = os.path.dirname(os.path.realpath(__file__))
        mujoco_env.MujocoEnv.__init__(self, '%s/assets/cartpole.xml' % dir_path, 2)
        self.horizon = 200
        low = np.array([-3, -5, -6, -20]).astype(np.float32)
        self.observation_space = spaces.Box(low=low, high=-low)

    def step(self, a):
        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()

        cost_lscale = PETSCartpoleEnv.PENDULUM_LENGTH
        reward = np.exp(
            -np.sum(np.square(self._get_ee_pos(ob) - np.array([0.0, PETSCartpoleEnv.PENDULUM_LENGTH]))) / (cost_lscale ** 2)
        )
        reward -= 0.01 * np.sum(np.square(a))

        done = False
        return ob, reward, done, {}

    def reset_model(self):
        qpos = self.init_qpos + np.random.normal(0, 0.1, np.shape(self.init_qpos))
        qvel = self.init_qvel + np.random.normal(0, 0.1, np.shape(self.init_qvel))
        self.set_state(qpos, qvel)
        return self._get_obs()

    def reset(self, obs=None):
        if obs is None:
            return super().reset()
        else:
            out = super().reset()
            qpos = obs[:len(self.init_qpos)]
            qvel = obs[len(self.init_qvel):]
            self.set_state(qpos, qvel)
            new_obs = self._get_obs()
            assert np.allclose(new_obs, obs)
            return new_obs


    def _get_obs(self):
        return np.concatenate([self.sim.data.qpos, self.sim.data.qvel]).ravel()

    @staticmethod
    def _get_ee_pos(x):
        x0, theta = x[0], x[1]
        return np.array([
            x0 - PETSCartpoleEnv.PENDULUM_LENGTH * np.sin(theta),
            -PETSCartpoleEnv.PENDULUM_LENGTH * np.cos(theta)
        ])

    def viewer_setup(self):
        v = self.viewer
        v.cam.trackbodyid = 0
        v.cam.distance = v.model.stat.extent


def cartpole_reward(x, y):
    '''
    x is state, action concatentated
    y is next_state  - state, (TODO: confirm)
    '''
    obs_dim = PETSCartpoleEnv.OBSERVATION_DIM
    next_obs = y
    action = x[obs_dim:]
    cost_lscale = PETSCartpoleEnv.PENDULUM_LENGTH
    reward = np.exp(
        -np.sum(np.square(PETSCartpoleEnv._get_ee_pos(next_obs) - np.array([0.0, PETSCartpoleEnv.PENDULUM_LENGTH]))) / (cost_lscale ** 2)
    )
    reward -= 0.01 * np.sum(np.square(action))
    return reward


def test_cartpole():
    env = PETSCartpoleEnv()
    n_tests = 10
    for _ in range(n_tests):
        obs = env.reset()
        action = env.action_space.sample()
        next_obs, rew, done, info = env.step(action)
        x = np.concatenate([obs, action])
        other_rew = cartpole_reward(x, next_obs)
        assert np.allclose(rew, other_rew)
        new_obs = env.reset(obs)
        assert np.allclose(new_obs, obs)
    print("passed")

if __name__ == '__main__':
    test_cartpole()
