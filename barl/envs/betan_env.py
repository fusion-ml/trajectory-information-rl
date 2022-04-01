from pathlib import Path
import numpy as np
from omegaconf import OmegaConf
from fusion_control.envs.fusion_env import FusionEnv
from fusion_control.envs.rewards import TrackingReward
from dynamics_toolbox.utils.storage.model_storage import load_model_from_log_dir
from fusion_control.envs.target_distributions import UniformTargetDistribution


class BetanRotationEnv(FusionEnv):
    def __init__(self):
        # TODO: fix path
        dynamics_model = load_model_from_log_dir(
            "/zfsauton/project/public/ichar/FusionControl/models/dynamics/scalar_simplex/100ms/train/2022-02-03"
        )
        rew_fn = TrackingReward(
            track_signals=["betan_EFIT01", "rotation_0"], track_coefficients=[1, 1]
        )
        target_distribution = UniformTargetDistribution(
            target_lows=[-0.17593358763413988, -0.3475464599161612],
            target_highs=[1.0847411701303655, 1.3775724083641652],
        )

        states_in_obs = [
            "betan_EFIT01",
            "betan_EFIT01_velocity",
            "rotation_0",
            "rotation_0_velocity",
        ]
        actuators_in_obs = ["pinj", "pinj_velocity", "tinj", "tinj_velocity"]
        action_space = ["pinj_velocity", "tinj_velocity"]
        """
        state_bounds = {"betan_EFIT01": [-1.722097423672676, 1.5961559265851974],
                        "betan_EFIT01_velocity": [-0.3916414469480514, 0.7564077183604244],
                        "rotation_0": [-0.9418659448623657, 3.0152017951011643],
                        "rotation_0_velocity": [-0.6269022941589355, 0.821212154626845]}
        actuator_bounds = {"pinj": [-1.4240562528371812, 1.6964053332805638],
                           "tinj": [-1.7613210678100586, 1.83820667564869],
                           "pinj_velocity": [-0.5567382872104645, 0.6844894513487817],
                           "tinj_velocity": [-0.697502514719963, 0.7395545974373825]}
        """
        state_bounds = load_bounds(
            Path(__file__).parent.resolve() / "fusion_cfg" / "state_bounds.yaml"
        )
        actuator_bounds = load_bounds(
            Path(__file__).parent.resolve() / "fusion_cfg" / "actuator_bounds.yaml"
        )
        self.horizon = 10
        super().__init__(
            dynamics_model=dynamics_model,
            reward_function=rew_fn,
            target_distribution=target_distribution,
            # TODO: fix this paht
            data_path="/zfsauton/project/public/ichar/FusionControl/data/scalar_data/100ms",
            states_in_obs=states_in_obs,
            actuators_in_obs=actuators_in_obs,
            action_space=action_space,
            actuator_bounds=actuator_bounds,
            state_bounds=state_bounds,
            max_horizon=self.horizon,
        )


def load_bounds(path):
    conf = OmegaConf.load(path)
    return conf


def betan_rotation_reward(x, next_obs):
    # TODO get these indices right
    betan_target = next_obs[..., 0]
    rot_target = next_obs[..., 2]
    betan = next_obs[..., 8]
    rot = next_obs[..., 9]
    return -1 * (np.abs(betan_target - betan) + np.abs(rot_target - rot))


if __name__ == "__main__":
    env = BetanRotationEnv()
    obs = env.reset()
    for i in range(env.horizon):
        action = env.action_space.sample()
        next_obs, rew, done, info = env.step(action)
        x = np.concatenate([obs, action])
        rew_hat = betan_rotation_reward(x, next_obs)
        assert np.allclose(rew_hat, rew)
