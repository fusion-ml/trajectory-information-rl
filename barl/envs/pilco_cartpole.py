"""
Cart pole swing-up: Identical version to PILCO V0.9
"""

import logging
import math
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import tensorflow as tf

logger = logging.getLogger(__name__)


def angle_normalize(x):
    return ((x + np.pi) % (2 * np.pi)) - np.pi


class CartPoleSwingUpEnv(gym.Env):
    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 50}
    OBSERVATION_DIM = 4
    POLE_LENGTH = 0.6

    def __init__(self, use_trig=False):
        self.use_trig = use_trig
        self.g = 9.82  # gravity
        self.m_c = 0.5  # cart mass
        self.m_p = 0.5  # pendulum mass
        self.total_m = self.m_p + self.m_c
        self.l = CartPoleSwingUpEnv.POLE_LENGTH  # pole's length
        self.m_p_l = self.m_p * self.l
        self.force_mag = 10.0
        self.dt = 0.1  # seconds between state updates
        self.b = 0.1  # friction coefficient
        self.horizon = 25
        self.periodic_dimensions = [2]

        # Angle at which to fail the episode
        self.theta_threshold_radians = 12 * 2 * math.pi / 360
        self.x_threshold = 2.4

        high = np.array([10.0, 10.0, 3.14159, 25.0])

        self.action_space = spaces.Box(-1, 1, shape=(1,))
        self.observation_space = spaces.Box(-high, high)
        if self.use_trig:
            high = np.array([10.0, 10.0, 1.0, 1.0, 25.0])
            self.observation_space = spaces.Box(-high, high)

        self._seed()
        self.viewer = None
        self.state = None

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        # Valid action
        action = np.clip(action, -1, 1)[0] * self.force_mag

        state = self.state
        x, x_dot, theta, theta_dot = state

        s = math.sin(theta)
        c = math.cos(theta)

        xdot_update = (
            -2 * self.m_p_l * (theta_dot**2) * s
            + 3 * self.m_p * self.g * s * c
            + 4 * action
            - 4 * self.b * x_dot
        ) / (4 * self.total_m - 3 * self.m_p * c**2)
        thetadot_update = (
            -3 * self.m_p_l * (theta_dot**2) * s * c
            + 6 * self.total_m * self.g * s
            + 6 * (action - self.b * x_dot) * c
        ) / (4 * self.l * self.total_m - 3 * self.m_p_l * c**2)
        x = x + x_dot * self.dt
        unnorm_theta = theta + theta_dot * self.dt
        theta = angle_normalize(unnorm_theta)
        x_dot = x_dot + xdot_update * self.dt
        theta_dot = theta_dot + thetadot_update * self.dt

        delta_s = np.array([x, x_dot, unnorm_theta, theta_dot]) - np.array(state)
        self.state = (x, x_dot, theta, theta_dot)

        # compute costs - saturation cost
        goal = np.array([0.0, self.l])
        pole_x = self.l * np.sin(theta)
        pole_y = self.l * np.cos(theta)
        position = np.array([self.state[0] + pole_x, pole_y])
        squared_distance = np.sum((position - goal) ** 2)
        squared_sigma = 0.25**2
        costs = 1 - np.exp(-0.5 * squared_distance / squared_sigma)

        return self.get_obs(), -costs, False, {"delta_obs": delta_s}

    def get_obs(self):
        if self.use_trig:
            return np.array(
                [
                    self.state[0],
                    self.state[1],
                    np.sin(self.state[2]),
                    np.cos(self.state[2]),
                    self.state[3],
                ]
            )
        else:
            return np.array(self.state)

    def reset(self, obs=None):
        # self.state = self.np_random.normal(loc=np.array([0.0, 0.0, 30*(2*np.pi)/360, 0.0]), scale=np.array([0.0, 0.0, 0.0, 0.0]))
        if obs is None:
            self.state = self.np_random.normal(
                loc=np.array([0.0, 0.0, np.pi, 0.0]),
                scale=np.array([0.02, 0.02, 0.02, 0.02]),
            )
        else:
            assert (
                not self.use_trig
            ), f"can't use trig if you are going to have generative access"
            self.state = obs
        self.state[2] = angle_normalize(self.state[2])
        return self.get_obs()

    def render(self):
        self._render()

    def _render(self, mode="human", close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        screen_width = 600
        screen_height = 400

        world_width = 5  # max visible position of cart
        scale = screen_width / world_width
        carty = 200  # TOP OF CART
        polewidth = 6.0
        polelen = scale * self.l  # 0.6 or self.l
        cartwidth = 40.0
        cartheight = 20.0

        if self.viewer is None:
            from gym.envs.classic_control import rendering

            self.viewer = rendering.Viewer(screen_width, screen_height)

            l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2

            cart = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            self.carttrans = rendering.Transform()
            cart.add_attr(self.carttrans)
            cart.set_color(1, 0, 0)
            self.viewer.add_geom(cart)

            l, r, t, b = (
                -polewidth / 2,
                polewidth / 2,
                polelen - polewidth / 2,
                -polewidth / 2,
            )
            pole = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            pole.set_color(0, 0, 1)
            self.poletrans = rendering.Transform(translation=(0, 0))
            pole.add_attr(self.poletrans)
            pole.add_attr(self.carttrans)
            self.viewer.add_geom(pole)

            self.axle = rendering.make_circle(polewidth / 2)
            self.axle.add_attr(self.poletrans)
            self.axle.add_attr(self.carttrans)
            self.axle.set_color(0.1, 1, 1)
            self.viewer.add_geom(self.axle)

            # Make another circle on the top of the pole
            self.pole_bob = rendering.make_circle(polewidth / 2)
            self.pole_bob_trans = rendering.Transform()
            self.pole_bob.add_attr(self.pole_bob_trans)
            self.pole_bob.add_attr(self.poletrans)
            self.pole_bob.add_attr(self.carttrans)
            self.pole_bob.set_color(0, 0, 0)
            self.viewer.add_geom(self.pole_bob)

            self.wheel_l = rendering.make_circle(cartheight / 4)
            self.wheel_r = rendering.make_circle(cartheight / 4)
            self.wheeltrans_l = rendering.Transform(
                translation=(-cartwidth / 2, -cartheight / 2)
            )
            self.wheeltrans_r = rendering.Transform(
                translation=(cartwidth / 2, -cartheight / 2)
            )
            self.wheel_l.add_attr(self.wheeltrans_l)
            self.wheel_l.add_attr(self.carttrans)
            self.wheel_r.add_attr(self.wheeltrans_r)
            self.wheel_r.add_attr(self.carttrans)
            self.wheel_l.set_color(0, 0, 0)  # Black, (B, G, R)
            self.wheel_r.set_color(0, 0, 0)  # Black, (B, G, R)
            self.viewer.add_geom(self.wheel_l)
            self.viewer.add_geom(self.wheel_r)

            self.track = rendering.Line(
                (0, carty - cartheight / 2 - cartheight / 4),
                (screen_width, carty - cartheight / 2 - cartheight / 4),
            )
            self.track.set_color(0, 0, 0)
            self.viewer.add_geom(self.track)

        if self.state is None:
            return None

        x = self.state
        cartx = x[0] * scale + screen_width / 2.0  # MIDDLE OF CART
        self.carttrans.set_translation(cartx, carty)
        self.poletrans.set_rotation(x[2])
        self.pole_bob_trans.set_translation(
            -self.l * np.sin(x[2]), self.l * np.cos(x[2])
        )

        return self.viewer.render(return_rgb_array=mode == "rgb_array")


def get_pole_pos(x):
    xpos = x[..., 0]
    theta = x[..., 2]
    pole_x = CartPoleSwingUpEnv.POLE_LENGTH * np.sin(theta)
    pole_y = CartPoleSwingUpEnv.POLE_LENGTH * np.cos(theta)
    position = np.array([xpos + pole_x, pole_y]).T
    return position


def pilco_cartpole_reward(x, next_obs):
    position = get_pole_pos(next_obs)
    goal = np.array([0.0, CartPoleSwingUpEnv.POLE_LENGTH])
    squared_distance = np.sum((position - goal) ** 2, axis=-1)
    squared_sigma = 0.25**2
    costs = 1 - np.exp(-0.5 * squared_distance / squared_sigma)
    return -costs


def tf_get_pole_pos(x):
    xpos = x[..., 0]
    theta = x[..., 2]
    pole_x = CartPoleSwingUpEnv.POLE_LENGTH * tf.sin(theta)
    pole_y = CartPoleSwingUpEnv.POLE_LENGTH * tf.cos(theta)
    position = tf.cast(tf.stack([xpos + pole_x, pole_y], axis=-1), tf.float32)
    return position


def tf_pilco_cartpole_reward(x, next_obs):
    position = tf_get_pole_pos(next_obs)
    goal = tf.constant([0.0, CartPoleSwingUpEnv.POLE_LENGTH], dtype=tf.float32)
    squared_distance = tf.reduce_sum((position - goal) ** 2, axis=-1)
    squared_sigma = 0.25**2
    costs = 1 - tf.exp(-0.5 * squared_distance / squared_sigma)
    return -costs


def test_cartpole():
    env = CartPoleSwingUpEnv()
    n_tests = 100
    for _ in range(n_tests):
        obs = env.reset()
        action = env.action_space.sample()
        next_obs, rew, done, info = env.step(action)
        x = np.concatenate([obs, action])
        other_rew = pilco_cartpole_reward(x, next_obs)
        tf_rew = tf_pilco_cartpole_reward(x, next_obs)
        assert np.allclose(other_rew, tf_rew)
        assert np.allclose(rew, other_rew)
        new_obs = env.reset(obs)
        assert np.allclose(new_obs, obs)
    done = False
    env.reset()
    for _ in range(env.horizon):
        action = env.action_space.sample()
        n, r, done, info = env.step(action)
        if done:
            break
    print("passed")


if __name__ == "__main__":
    test_cartpole()
