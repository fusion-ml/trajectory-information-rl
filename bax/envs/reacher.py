import os
import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env


class BACReacherEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        utils.EzPickle.__init__(self)
        self.horizon = 50
        self.periodic_dimensions = []
        dir_path = os.path.dirname(os.path.realpath(__file__))
        mujoco_env.MujocoEnv.__init__(self, "%s/assets/reacher.xml" % dir_path, 2)

    def step(self, a):
        vec = self.get_body_com("fingertip") - self.get_body_com("target")
        vec = vec[:2]
        reward_dist = -np.linalg.norm(vec)
        reward_ctrl = -np.square(a).sum()
        reward = reward_dist + reward_ctrl
        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        done = False
        return ob, reward, done, dict(reward_dist=reward_dist, reward_ctrl=reward_ctrl)

    def reset(self, obs=None):
        old_obs = super().reset()
        if obs is None:
            return old_obs
        full_obs = np.concatenate([obs[:-2], np.zeros(2)])
        qpos = full_obs[:len(self.init_qpos)]
        qvel = full_obs[len(self.init_qpos):]
        self.set_state(qpos, qvel)
        check_obs = self._get_obs()
        # assert np.allclose(check_obs, obs, atol=1e-3), f"Obs: {obs} not equal to check_obs {check_obs}"
        return check_obs

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 0

    def reset_model(self):
        qpos = (
            self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nq)
            + self.init_qpos
        )
        while True:
            self.goal = self.np_random.uniform(low=-0.2, high=0.2, size=2)
            if np.linalg.norm(self.goal) < 0.2:
                break
        qpos[-2:] = self.goal
        qvel = self.init_qvel + self.np_random.uniform(
            low=-0.005, high=0.005, size=self.model.nv
        )
        qvel[-2:] = 0
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_obs(self):
        # theta = self.sim.data.qpos.flat[:2]
        vec = self.get_body_com("fingertip") - self.get_body_com("target")
        vec = vec[:2]
        return np.concatenate(
            [
                # np.cos(theta),
                # np.sin(theta),
                self.sim.data.qpos.flat,
                self.sim.data.qvel.flat[:2],
                vec,
            ]
        )


def reacher_reward(x, next_obs):
    action_dim = 2
    start_obs = x[..., :-action_dim]
    vec = start_obs[..., -2:]
    action = x[..., -action_dim:]
    reward_dist = -np.linalg.norm(vec, axis=-1)
    reward_ctrl = -np.square(action).sum(axis=-1)
    reward = reward_dist + reward_ctrl
    return reward


if __name__ == "__main__":
    # we have an increased ATOL because the COM part of the state is the solution
    # to some kind of FK problem and can have numerical error
    env = BACReacherEnv()
    og_obs = env.reset()
    obs = og_obs
    done = False
    for _ in range(env.horizon):
        action = env.action_space.sample()
        next_obs, rew, done, info = env.step(action)
        x = np.concatenate([obs, action])
        other_rew = reacher_reward(x, next_obs)
        assert np.allclose(rew, other_rew, atol=1e-3), f"{rew=}, {other_rew=}"
        obs = next_obs
        new_obs = env.reset(obs)
        assert np.allclose(new_obs, obs, atol=1e-3)
    # test reset to point
    env.reset(og_obs)
