import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from os import path


class PendulumEnv(gym.Env):
    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 30}

    def __init__(
        self, g=10.0, seed=None, tight_start=False, medium_start=False, test_case=False
    ):

        # Set gym env seed
        assert not (tight_start and medium_start)
        self.seed(seed)

        self.max_speed = 8
        self.max_torque = 2.0
        self.dt = 0.05
        self.g = g
        self.m = 1.0
        self.l = 1.0
        self.viewer = None
        self.horizon = 200
        self.periodic_dimensions = [0]
        self.tight_start = tight_start
        self.medium_start = medium_start
        if test_case:
            self.tight_start = True
            self.medium_start = False
            self.horizon = 20

        high = np.array([np.pi, self.max_speed], dtype=np.float32)
        self.action_space = spaces.Box(
            low=-self.max_torque, high=self.max_torque, shape=(1,), dtype=np.float32
        )
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, u):
        th, thdot = self.state  # th := theta

        g = self.g
        m = self.m
        l = self.l
        dt = self.dt

        u = np.clip(u, -self.max_torque, self.max_torque)[0]
        self.last_u = u  # for rendering

        newthdot = (
            thdot
            + (-3 * g / (2 * l) * np.sin(th + np.pi) + 3.0 / (m * l**2) * u) * dt
        )
        unnorm_newth = th + newthdot * dt
        newth = angle_normalize(unnorm_newth)
        newthdot = np.clip(newthdot, -self.max_speed, self.max_speed)

        costs = angle_normalize(newth) ** 2 + 0.1 * newthdot**2 + 0.001 * (u**2)
        delta_s = np.array([unnorm_newth - th, newthdot - thdot])

        self.state = np.array([newth, newthdot])
        done = False
        return self._get_obs(), -costs, done, {"delta_obs": delta_s}

    def reset(self, obs=None):
        high = np.array([np.pi, 1])
        if obs is None:
            if self.tight_start:
                self.state = self.np_random.uniform(
                    low=[-0.35, -0.9], high=[-0.05, -0.6]
                )
            elif self.medium_start:
                self.state = self.np_random.uniform(low=[-3, -1], high=[-1, 1])
            else:
                self.state = self.np_random.uniform(low=-high, high=high)
        else:
            self.state = obs
        self.last_u = None
        return self.state

    def _get_obs(self):
        theta, thetadot = self.state
        return self.state

    def render(self, mode="human"):
        if self.viewer is None:
            from gym.envs.classic_control import rendering

            self.viewer = rendering.Viewer(500, 500)
            self.viewer.set_bounds(-2.2, 2.2, -2.2, 2.2)
            rod = rendering.make_capsule(1, 0.2)
            rod.set_color(0.8, 0.3, 0.3)
            self.pole_transform = rendering.Transform()
            rod.add_attr(self.pole_transform)
            self.viewer.add_geom(rod)
            axle = rendering.make_circle(0.05)
            axle.set_color(0, 0, 0)
            self.viewer.add_geom(axle)
            fname = path.join(path.dirname(__file__), "assets/clockwise.png")
            self.img = rendering.Image(fname, 1.0, 1.0)
            self.imgtrans = rendering.Transform()
            self.img.add_attr(self.imgtrans)

        self.viewer.add_onetime(self.img)
        self.pole_transform.set_rotation(self.state[0] + np.pi / 2)
        if self.last_u:
            self.imgtrans.scale = (-self.last_u / 2, np.abs(self.last_u) / 2)

        return self.viewer.render(return_rgb_array=mode == "rgb_array")

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None


def angle_normalize(x):
    return ((x + np.pi) % (2 * np.pi)) - np.pi


def pendulum_reward(x, next_obs):
    th = next_obs[..., 0]
    thdot = next_obs[..., 1]
    u = x[..., 2]
    costs = angle_normalize(th) ** 2 + 0.1 * thdot**2 + 0.001 * (u**2)
    return -costs
