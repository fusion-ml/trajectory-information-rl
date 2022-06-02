import numpy as np
import os
from gym import utils, spaces
from gym.envs.mujoco import mujoco_env


DEFAULT_CAMERA_CONFIG = {
    "distance": 4.0,
}


class HalfCheetahEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(
        self,
        xml_file="half_cheetah.xml",
        forward_reward_weight=1.0,
        ctrl_cost_weight=0.1,
        reset_noise_scale=0.1,
        exclude_current_positions_from_observation=False,
    ):
        utils.EzPickle.__init__(**locals())

        self._forward_reward_weight = forward_reward_weight

        self._ctrl_cost_weight = ctrl_cost_weight

        self._reset_noise_scale = reset_noise_scale

        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation
        )

        xml_file = (
            f"{os.path.dirname(os.path.realpath(__file__))}/assets/half_cheetah.xml"
        )

        mujoco_env.MujocoEnv.__init__(self, xml_file, 5)
        self.horizon = 200
        self.periodic_dimensions = []
        low = np.ones(18) * -1000
        high = -low
        self.observation_space = spaces.Box(low=low, high=high)

    def control_cost(self, action):
        control_cost = self._ctrl_cost_weight * np.sum(np.square(action))
        return control_cost

    def step(self, action):
        x_position_before = self.sim.data.qpos[0]
        old_obs = self._get_obs()
        self.do_simulation(action, self.frame_skip)
        x_position_after = self.sim.data.qpos[0]
        x_velocity = (x_position_after - x_position_before) / self.dt

        ctrl_cost = self.control_cost(action)

        forward_reward = self._forward_reward_weight * x_velocity

        observation = self._get_obs()
        reward = forward_reward - ctrl_cost
        done = False
        delta_obs = observation - old_obs
        info = {
            "x_position": x_position_after,
            "x_velocity": x_velocity,
            "reward_run": forward_reward,
            "reward_ctrl": -ctrl_cost,
            "delta_obs": delta_obs,
        }

        return observation, reward, done, info

    def _get_obs(self):
        position = self.sim.data.qpos.flat.copy()
        velocity = self.sim.data.qvel.flat.copy()

        if self._exclude_current_positions_from_observation:
            position = position[1:]

        observation = np.concatenate((position, velocity)).ravel()
        return observation

    def reset(self, obs=None):
        old_obs = super().reset()
        if obs is None:
            return old_obs
        qpos = obs[: len(self.init_qpos)]
        qvel = obs[len(self.init_qpos) :]
        self.set_state(qpos, qvel)
        check_obs = self._get_obs()
        assert np.allclose(check_obs, obs), f"check_obs={check_obs}, obs={obs}"
        return obs

    def reset_model(self):
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq
        )
        qvel = self.init_qvel + self._reset_noise_scale * self.np_random.randn(
            self.model.nv
        )

        self.set_state(qpos, qvel)

        observation = self._get_obs()
        return observation

    def viewer_setup(self):
        for key, value in DEFAULT_CAMERA_CONFIG.items():
            if isinstance(value, np.ndarray):
                getattr(self.viewer.cam, key)[:] = value
            else:
                setattr(self.viewer.cam, key, value)


def half_cheetah_reward(x, next_obs):
    ctrl_cost_weight = 0.1
    forward_reward_weight = 1.0
    action_dim = 6
    action = x[..., -action_dim:]
    ctrl_cost = ctrl_cost_weight * np.sum(np.square(action), axis=-1)
    x_position_before = x[..., 0]
    x_position_after = next_obs[..., 0]
    dt = 0.05
    x_vel = (x_position_after - x_position_before) / dt
    forward_reward = forward_reward_weight * x_vel
    return forward_reward - ctrl_cost


if __name__ == "__main__":
    env = HalfCheetahEnv()
    print(
        f"env.observation_space={env.observation_space}, env.action_space={env.action_space}"
    )
    og_obs = env.reset()
    obs = og_obs
    done = False
    for _ in range(env.horizon):
        action = env.action_space.sample()
        next_obs, rew, done, info = env.step(action)
        x = np.concatenate([obs, action])
        other_rew = half_cheetah_reward(x, next_obs)
        assert np.allclose(rew, other_rew), f"rew={rew}, other_rew={other_rew}"
        obs = next_obs
        new_obs = env.reset(obs)
        assert np.allclose(new_obs, obs), f"new_obs={new_obs}, obs={obs}"
    # test reset to point
    env.reset(og_obs)
