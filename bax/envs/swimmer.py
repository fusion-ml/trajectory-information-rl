import numpy as np
from gym.envs.mujoco import mujoco_env
import os
from gym import utils


DEFAULT_CAMERA_CONFIG = {}


class BACSwimmerEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(
        self,
        forward_reward_weight=1.0,
        ctrl_cost_weight=1e-4,
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

        dir_path = os.path.dirname(os.path.realpath(__file__))
        self.horizon = 1000
        mujoco_env.MujocoEnv.__init__(self, '%s/assets/swimmer.xml' % dir_path, 4)
        # TODO: set real obs spaces

    def control_cost(self, action):
        control_cost = self._ctrl_cost_weight * np.sum(np.square(action))
        return control_cost

    def step(self, action):
        xy_position_before = self.sim.data.qpos[0:2].copy()
        self.do_simulation(action, self.frame_skip)
        xy_position_after = self.sim.data.qpos[0:2].copy()

        xy_velocity = (xy_position_after - xy_position_before) / self.dt
        x_velocity, y_velocity = xy_velocity

        forward_reward = self._forward_reward_weight * x_velocity

        ctrl_cost = self.control_cost(action)

        observation = self._get_obs()
        reward = forward_reward - ctrl_cost
        done = False
        info = {
            "reward_fwd": forward_reward,
            "reward_ctrl": -ctrl_cost,
            "x_position": xy_position_after[0],
            "y_position": xy_position_after[1],
            "distance_from_origin": np.linalg.norm(xy_position_after, ord=2),
            "x_velocity": x_velocity,
            "y_velocity": y_velocity,
            "forward_reward": forward_reward,
        }

        return observation, reward, done, info

    def _get_obs(self):
        position = self.sim.data.qpos.flat.copy()
        velocity = self.sim.data.qvel.flat.copy()

        if self._exclude_current_positions_from_observation:
            position = position[2:]

        observation = np.concatenate([position, velocity]).ravel()
        return observation

    def reset(self, obs=None):
        old_obs = super().reset()
        if obs is None:
            return old_obs
        if self._exclude_current_positions_from_observation:
            position = np.zeros(2)
            full_obs = np.concatenate([position, obs])
        else:
            full_obs = obs
        qpos = full_obs[:len(self.init_qpos)]
        qvel = full_obs[len(self.init_qpos):]
        self.set_state(qpos, qvel)
        check_obs = self._get_obs()
        assert np.allclose(check_obs, obs)
        return check_obs

    def reset_model(self):
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq
        )
        qvel = self.init_qvel + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nv
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


def swimmer_reward(x, next_obs):
    dt = 0.04
    forward_reward_weight = 1.0
    action_dim = 2
    control_cost_weight = 1e-4
    pos_before = x[..., :2]
    pos_after = next_obs[..., :2]
    xy_velocity = (pos_after - pos_before) / dt
    x_velocity = xy_velocity[..., 0]

    forward_reward = forward_reward_weight * x_velocity

    action = x[..., -action_dim:]
    control_cost = np.sum(np.square(action), axis=-1) * control_cost_weight
    return forward_reward - control_cost


if __name__ == "__main__":
    env = BACSwimmerEnv()
    print(f"{env.dt=}")
    og_obs = env.reset()
    obs = og_obs
    done = False
    for _ in range(env.horizon):
        action = env.action_space.sample()
        next_obs, rew, done, info = env.step(action)
        x = np.concatenate([obs, action])
        other_rew = swimmer_reward(x, next_obs)
        assert np.allclose(rew, other_rew)
        obs = next_obs
        new_obs = env.reset(obs)
        assert np.allclose(new_obs, obs)
    # test reset to point
    env.reset(og_obs)
