"""
Environment for beta tracking.
"""
import pickle as pkl

import gym
import numpy as np

from dynamics_toolbox.utils.storage.qdata import load_from_hdf5
from dynamics_toolbox.utils.storage.model_storage import load_model_from_log_dir
from rlkit.envs.env_model import EnvModel
from rlkit.data_management.mb_start_selectors import StartSelector


DEFAULT_STATE_SPACE = (
    "betan_EFIT01",
    "betan_EFIT01_prev_delta",
    "pinj",
    "pinj_prev_delta",
)
BETA_TAG = "betan_EFIT01"
DEFAULT_MODEL_PATH = "/zfsauton/project/public/ichar/BetaTracking/models/mlp_beta_only"
DEFAULT_MODEL_INFO_PATH = (
    "/zfsauton/project/public/ichar/BetaTracking/data/beta_only/beta_only_info.pkl"
)
DEFAULT_START_DATA_PATH = (
    "/zfsauton/project/public/ichar/BetaTracking/data/beta_only/beta_only_data.hdf5"
)


class BetaTrackingModelEnv(EnvModel):
    """Model environment for beta tracking."""

    def __init__(
        self,
        model_path,
        info_dict_path,
        target=2,
        # If target distribution is specified this becomes a multi-goal probelm.
        # Then the goal betan is put into the state as the last observation.
        target_distribution=None,
        unnormalize_obs=False,
        # These default bounds are roughly the 0.05-0.95 interval.
        scaled_pinj_bounds=(-1, 1.5),
        # scaled_pinj_dt_bounds=None,
        scaled_pinj_dt_bounds=(-0.5, 0.5),
        action_is_change=True,
        # It is assumed state space will be ordered like ...
        # (signalX, signalX_prev_delta, ...., pinj, pinj_prev_delta)
        state_space=DEFAULT_STATE_SPACE,
        gpu_num=None,
        # Automatically set to True if target distribution is specified.
        include_target_in_state_space=False,
        full_state_space=False,
        **kwargs,
    ):
        self.model = load_model_from_log_dir(model_path)
        if gpu_num is not None:
            self.model.to(f"cuda:{gpu_num}")
        with open(info_dict_path, "rb") as f:
            self.info = pkl.load(f)
        self.unnormalize_obs = unnormalize_obs
        self.target = target
        self.target_distribution = target_distribution
        self.scaled_pinj_bounds = scaled_pinj_bounds
        self.scaled_pinj_dt_bounds = scaled_pinj_dt_bounds
        self.action_is_change = action_is_change
        if scaled_pinj_dt_bounds is None and action_is_change:
            raise ValueError("Pinj dt bounds need to be provided if action is change.")
        if full_state_space:
            self.state_space = self.info["x_columns"][:-1]
        else:
            self.state_space = state_space
        self.state_space = state_space
        self.include_target_in_state_space = include_target_in_state_space or (
            target_distribution is not None
        )
        self.obs_dim = len(state_space) + self.include_target_in_state_space
        self._observation_space = gym.spaces.Box(
            low=-3 * np.ones(self.obs_dim),
            high=3 * np.ones(self.obs_dim),
            dtype=np.float32,
        )
        self._action_space = gym.spaces.Box(
            low=-1, high=1, dtype=np.float32, shape=(1,)
        )

    def unroll(self, start_states, policy, horizon, actions=None):
        """Unroll for multiple trajectories at once.
        Args:
            start_states: The start states to unroll at as ndarray
                w shape (num_starts, obs_dim).
            policy: Policy to take actions.
            horizon: How long to rollout for.
            actions: The actions to use to unroll.

        Returns:
            * obs ndarray of (horizon + 1, num_starts, obs_dim)
            * actions ndarray of (horizon, num_starts, act_dim)
            * rewards ndarray of (horizon, num_starts)
            * terminals ndarray of (horizon, num_starts)
            * env_info mapping from str -> ndarray
            * actor_info mapping str -> ndarray
        """
        should_call_policy = actions is None
        # Init the datastructures.
        states = np.zeros((horizon + 1, start_states.shape[0], start_states.shape[1]))
        obs = np.zeros((horizon + 1, start_states.shape[0], self.obs_dim))
        if self.target_distribution is not None:
            obs[..., -1] = np.tile(
                self.target_distribution(start_states.shape[0]), horizon + 1
            ).reshape(horizon + 1, start_states.shape[0])
        elif self.include_target_in_state_space:
            obs[..., -1] = self.target
        if actions is None:
            actions = np.zeros((horizon, start_states.shape[0], 1))
        states[0] = start_states
        first_obs = self._extract_obs(start_states)
        obs[0, :, : first_obs.shape[1]] = first_obs
        rewards = np.zeros((horizon, start_states.shape[0]))
        terminals = np.full((horizon, start_states.shape[0]), False)
        logpis = np.zeros((horizon, start_states.shape[0]))
        for hidx in range(horizon):
            # Get actions for each of the states.
            if should_call_policy:
                net_in = obs[hidx]
                acts, probs = policy.get_actions(net_in)
                acts = acts.flatten()
                actions[hidx] = acts.reshape(-1, 1)
                logpis[hidx] = probs
            else:
                acts = actions[hidx].flatten()
            # Roll all states forward.
            nxt_info = self.multi_step(states[hidx], acts)
            states[hidx + 1] = nxt_info["state"]
            obs[hidx + 1, :, : nxt_info["obs"].shape[1]] = nxt_info["obs"]
            if self.target_distribution is None:
                targets = None
            else:
                targets = obs[hidx, :, -1]
            rewards[hidx] = self._compute_rew(nxt_info["state"], targets)
        env_infos = {
            "targets": np.ones((horizon, start_states.shape[0])) * self.target
            if self.target_distribution is None
            else obs[..., -1]
        }
        agent_infos = {"logpi": logpis}
        return obs, actions, rewards, terminals, env_infos, agent_infos

    def multi_step(self, states, actions):
        pid_nxt_delta = self._action_to_normd_pinj(states, actions)
        model_input = np.hstack([states, pid_nxt_delta.reshape(-1, 1)])
        signal_deltas = self.model.predict(model_input)[0]
        nxt_states = np.array(states)
        # Set signal observations based on predicted velocities.
        nxt_states[:, [i * 2 for i in range(signal_deltas.shape[1])]] += signal_deltas
        # Set predicted signal velocities.
        nxt_states[
            :, [i * 2 + 1 for i in range(signal_deltas.shape[1])]
        ] = signal_deltas
        nxt_states[:, -2] += pid_nxt_delta
        nxt_states[:, -1] = pid_nxt_delta
        obs = self._extract_obs(nxt_states)
        return {
            "state": nxt_states,
            "obs": obs,
        }

    def _action_to_normd_pinj(self, states, actions):
        curr_pinjs = states[:, -2]
        actions = (actions + 1) / 2
        if self.action_is_change:
            pinj_change = (
                actions
                * (self.scaled_pinj_dt_bounds[1] - self.scaled_pinj_dt_bounds[0])
                + self.scaled_pinj_dt_bounds[0]
            )
            return np.clip(
                pinj_change,
                np.clip(
                    self.scaled_pinj_bounds[0] - curr_pinjs,
                    np.ones(len(curr_pinjs)) * self.scaled_pinj_dt_bounds[0],
                    np.zeros(len(curr_pinjs)),
                ),
                np.clip(
                    self.scaled_pinj_bounds[1] - curr_pinjs,
                    np.zeros(len(curr_pinjs)),
                    np.ones(len(curr_pinjs)) * self.scaled_pinj_dt_bounds[1],
                ),
            )
        else:
            pinj_targets = (
                actions * (self.scaled_pinj_bounds[1] - self.scaled_pinj_bounds[0])
                + self.scaled_pinj_bounds[0]
            )
            pinj_change = pinj_targets - curr_pinjs
            if self.scaled_pinj_dt_bounds is not None:
                pinj_change = np.clip(
                    pinj_change,
                    self.scaled_pinj_dt_bounds[0],
                    self.scaled_pinj_dt_bounds[1],
                )
            return pinj_change

    def _extract_obs(self, states):
        indices = [self.info["x_columns"].index(s) for s in self.state_space]
        obs = states[:, indices]
        if self.unnormalize_obs:
            for sidx, ss in enumerate(self.state_space):
                for k, v in self.info["normalization_dict"].items():
                    if k in ss:
                        obs[:, sidx] *= v["iqr"]
                        if "delta" not in ss:
                            obs[:, sidx] += v["median"]
        return obs

    def _compute_rew(self, states, targets=None):
        betas = states[:, self.info["x_columns"].index(BETA_TAG)]
        beta_dict = self.info["normalization_dict"][BETA_TAG]
        betas = betas * beta_dict["iqr"] + beta_dict["median"]
        if targets is None:
            return -1 * np.abs(betas - self.target)
        else:
            return -1 * np.abs(betas - targets)

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space


class BetaTrackingStartSelector(StartSelector):
    def __init__(self, data_path, test_shots=False, shuffle=True):
        data = load_from_hdf5(data_path)
        prefix = "te_" if test_shots else "tr_"
        curr_shot = data[prefix + "shotnums"][0]
        starts = [data[prefix + "x"][0]]
        for idx in range(1, len(data[prefix + "shotnums"])):
            if data[prefix + "shotnums"][idx] != curr_shot:
                curr_shot = data[prefix + "shotnums"][idx]
                starts.append(data[prefix + "x"][idx])
        self.starts = np.vstack(starts)[:, :-1]
        self.shuffle = shuffle

    def get_starts(self, num_starts: int) -> np.ndarray:
        """Get start states."""
        if self.shuffle:
            indices = np.random.randint(0, len(self.starts), num_starts)
            return self.starts[indices]
        return self.starts[:num_starts]


class BetaTrackingGymEnv(gym.Env):
    def __init__(
        self,
        model_path=DEFAULT_MODEL_PATH,
        model_info_path=DEFAULT_MODEL_INFO_PATH,
        start_data_path=DEFAULT_START_DATA_PATH,
        target=2,
        shuffle=True,
    ):
        self._model_env = BetaTrackingModelEnv(
            model_path,
            model_info_path,
            action_is_change=True,
            unnormalize_obs=False,
            target=target,
        )

        self._start_selector = BetaTrackingStartSelector(
            start_data_path, shuffle=shuffle, test_shots=False
        )

        self._state = None
        self._target = None
        if shuffle:
            self.horizon = 15
        else:
            self.horizon = 30
        self.periodic_dimensions = []

    def reset(self, obs=None):
        if obs is None:
            self._state = self._start_selector.get_starts(1)
        else:
            self._state = np.atleast_2d(obs)
        if self._model_env.target_distribution is not None:
            self._target = self._model_env.target_distribution(1)
        return self._model_env._extract_obs(self._state).flatten()

    def step(self, action):
        if not isinstance(action, np.ndarray):
            action = np.array([action])
        result = self._model_env.multi_step(self._state, action)
        self._state = result["state"]
        rew = self._model_env._compute_rew(self._state, targets=None)
        return result["obs"].flatten(), float(rew), False, {}

    @property
    def observation_space(self):
        return self._model_env.observation_space

    @property
    def action_space(self):
        return self._model_env.action_space


def beta_tracking_rew(x, next_obs, target=2):
    BETA_IDX = 0
    betas = next_obs[..., BETA_IDX]
    iqr = 0.8255070447921753
    median = 1.622602
    betas = betas * iqr + median
    return -1 * np.abs(betas - target)


def test_beta_tracking_env():
    num_trials = 50
    env = BetaTrackingGymEnv()
    returns = []
    for trial in range(num_trials):
        obs = env.reset()
        observations = []
        actions = []
        rewards = []
        next_observations = []
        for _ in range(env.horizon):
            action = env.action_space.sample()
            next_obs, rew, done, info = env.step(action)
            observations.append(obs)
            next_observations.append(next_obs)
            rewards.append(rew)
            actions.append(action)
            obs = next_obs
        observations = np.array(observations)
        action = np.array(actions)
        next_observations = np.array(next_observations)
        x = np.concatenate([observations, actions], axis=1)
        pred_rewards = beta_tracking_rew(x, next_observations)
        # test reward function
        assert np.allclose(pred_rewards, rewards)
        returns.append(np.sum(rewards))
    print(f"Random actions give {np.mean(returns)} return +- {np.std(returns)}")

    # test reset to state
    for obs in observations:
        reset_obs = env.reset(obs)
        assert np.allclose(reset_obs, obs)


if __name__ == "__main__":
    test_beta_tracking_env()
