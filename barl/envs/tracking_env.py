"""
A generic tracking environment.

Author: Ian Char
Date: 9/29/2021
"""
import pickle as pkl

import gym
import numpy as np

from dynamics_toolbox.utils.storage.qdata import load_from_hdf5
from dynamics_toolbox.utils.storage.model_storage import load_model_from_log_dir
from rlkit.envs.env_model import EnvModel
from rlkit.data_management.mb_start_selectors import StartSelector


BETA_ROTATION_OBS_SPACE = (
    "betan_EFIT01",
    "betan_EFIT01_prev_delta",
    "rotation",
    "rotation_prev_delta",
    "pinj",
    "pinj_prev_delta",
    "tinj",
    "tinj_prev_delta",
)
BETA_ROTATION_ACTION_SPACE = ("pinj_next_delta", "tinj_next_delta")
BETA_ROTATION_TARGET_LABELS = ("betan_EFIT01", "rotation")
# These default bounds are roughly the 0.05-0.95 interval.
SCALED_BETA_ROTATION_ACTION_BOUNDS = np.array([[-1, -1.65], [1.582, 1.45]])
SCALED_BETA_ROTATION_ACTION_DT_BOUNDS = np.array([[-0.48, -0.6], [0.583, 0.628]])
DEFAULT_MODEL_PATH = (
    "/zfsauton/project/public/ichar/BetaTracking/models/beta_rotation_mlp_limited"
)
DEFAULT_MODEL_INFO_PATH = "/zfsauton/project/public/ichar/BetaTracking/data/beta_rotation_data_limited/info.pkl"
DEFAULT_START_DATA_PATH = "/zfsauton/project/public/ichar/BetaTracking/data/beta_rotation_data_limited/data.hdf5"


class TrackingModelEnv(EnvModel):
    """Model environment for beta and rotation tracking."""

    def __init__(
        self,
        model_path,
        info_dict_path,
        target_distribution,
        should_unnormalize_obs=False,
        # If none for either bound, it is unbounded.
        scaled_action_bounds=SCALED_BETA_ROTATION_ACTION_BOUNDS,
        scaled_action_dt_bounds=SCALED_BETA_ROTATION_ACTION_DT_BOUNDS,
        action_is_change=True,
        obs_fields=BETA_ROTATION_OBS_SPACE,
        action_fiels=BETA_ROTATION_ACTION_SPACE,
        target_fields=BETA_ROTATION_TARGET_LABELS,
        gpu_num=None,
        full_obs=False,
        **kwargs,
    ):
        self.model = load_model_from_log_dir(model_path)
        if gpu_num is not None:
            self.model.to(f"cuda:{gpu_num}")
        with open(info_dict_path, "rb") as f:
            self.info = pkl.load(f)
        self.should_unnormalize_obs = should_unnormalize_obs
        self.scaled_action_bounds = scaled_action_bounds
        self.scaled_action_dt_bounds = scaled_action_dt_bounds
        self.action_is_change = action_is_change
        if full_obs:
            self.obs_fields = [
                s for s in self.info["x_columns"] if "next_delta" not in s
            ]
        else:
            self.obs_fields = obs_fields
        self.obs_fields = obs_fields
        self.action_fiels = action_fiels
        self.target_dim = len(target_distribution(1).flatten())
        self.obs_dim = len(obs_fields)  #  + self.target_dim
        self.act_dim = len(action_fiels)
        self.target_fields = target_fields
        self.target_distribution = self._wrap_target_dist(target_distribution)
        self._observation_space = gym.spaces.Box(
            low=-3 * np.ones(self.obs_dim),
            high=3 * np.ones(self.obs_dim),
            dtype=np.float32,
        )
        self._action_space = gym.spaces.Box(
            low=-1 * np.ones(self.act_dim),
            high=1 * np.ones(self.act_dim),
            dtype=np.float32,
        )
        # Figure out some indices and orderings ahead of time.
        self._state_fields = [
            s for s in self.info["x_columns"] if "next_delta" not in s
        ]
        self._act_current_idxs = [
            self._state_fields.index(get_signal(a)) for a in action_fiels
        ]
        self._act_prev_idxs = [
            self._state_fields.index(get_signal(a) + "_prev_delta")
            for a in action_fiels
        ]
        self._state_pred_curr = [
            self._state_fields.index(get_signal(s)) for s in self.info["y_columns"]
        ]
        self._state_pred_prev_delta = [
            self._state_fields.index(get_signal(s) + "_prev_delta")
            for s in self.info["y_columns"]
        ]
        self._obs_indices = [self._state_fields.index(s) for s in self.obs_fields]
        state_act_stack = self._state_fields + list(self.action_fiels)
        self._state_action_hstack_ordering = [
            state_act_stack.index(xc) for xc in self.info["x_columns"]
        ]
        # Figure out the statistics to do normalization ahead of time.
        self._obs_median = np.array(
            [
                0
                if "delta" in of
                else self.info["normalization_dict"][get_signal(of)]["median"]
                for of in self.obs_fields
            ]
        ).reshape(1, -1)
        # self._obs_median = np.array([
        #     0 if 'delta' in of
        #     else self.info['normalization_dict'][get_signal(of)]['median']
        #     for of in self.obs_fields + self.target_fields]).reshape(1, -1)
        self._obs_iqr = np.array(
            [
                self.info["normalization_dict"][get_signal(of)]["iqr"]
                for of in self.obs_fields
            ]
        ).reshape(1, -1)
        # self._obs_iqr = np.array([
        #     self.info['normalization_dict'][get_signal(of)]['iqr']
        #     for of in self.obs_fields + self.target_fields]).reshape(1, -1)

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
        targets = self.target_distribution(len(start_states))
        if actions is None:
            actions = np.zeros((horizon, start_states.shape[0], self.act_dim))
        states[0] = start_states
        first_obs = self.state_to_obs(start_states, targets)
        obs[0] = first_obs
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
            nxt_info = self.multi_step(states[hidx], acts, targets)
            states[hidx + 1] = nxt_info["state"]
            obs[hidx + 1] = nxt_info["obs"]
            rewards[hidx] = self.compute_rew(nxt_info["state"], targets)
        env_infos = {"targets": targets}
        agent_infos = {"logpi": logpis}
        return obs, actions, rewards, terminals, env_infos, agent_infos

    def multi_step(self, states, actions, targets=None):
        acts_next_delta = self.action_to_normd_signal(states, actions)
        model_input = np.hstack([states, acts_next_delta.reshape(-1, self.act_dim)])
        model_input = model_input[:, self._state_action_hstack_ordering]
        signal_deltas = self.model.predict(model_input)[0]
        nxt_states = np.array(states)
        # Set signal observations based on predicted velocities.
        nxt_states[:, self._state_pred_curr] += signal_deltas
        # Set predicted signal velocities.
        nxt_states[:, self._state_pred_prev_delta] = signal_deltas
        nxt_states[:, self._act_current_idxs] += acts_next_delta
        nxt_states[:, self._act_prev_idxs] = acts_next_delta
        # Put into correct observation.
        if targets is None:
            targets = np.zeros((len(states), self.target_dim))
        obs = self.state_to_obs(nxt_states, targets)
        return {
            "state": nxt_states,
            "obs": obs,
        }

    def action_to_normd_signal(self, states, actions):
        curr_acts = states[:, self._act_current_idxs]
        actions = (actions + 1) / 2
        if self.action_is_change:
            act_change = (
                actions
                * (
                    self.scaled_action_dt_bounds[[1]]
                    - self.scaled_action_dt_bounds[[0]]
                )
                + self.scaled_action_dt_bounds[[0]]
            )
            return np.clip(
                act_change,
                np.clip(
                    self.scaled_action_bounds[[0]] - curr_acts,
                    np.ones(curr_acts.shape) * self.scaled_action_dt_bounds[[0]],
                    np.zeros(curr_acts.shape),
                ),
                np.clip(
                    self.scaled_action_bounds[[1]] - curr_acts,
                    np.zeros(curr_acts.shape),
                    np.ones(curr_acts.shape) * self.scaled_action_dt_bounds[[1]],
                ),
            )
        else:
            act_targets = (
                actions
                * (self.scaled_action_bounds[[1]] - self.scaled_action_bounds[[0]])
                + self.scaled_action_bounds[[0]]
            )
            act_change = act_targets - curr_acts
            act_change = np.clip(
                act_change,
                self.scaled_action_dt_bounds[[0]],
                self.scaled_action_dt_bounds[[1]],
            )
            return act_change

    def state_to_obs(self, state, targets):
        # obs = np.hstack([state[:, self._obs_indices], targets])
        obs = state[:, self._obs_indices]
        if self.should_unnormalize_obs:
            obs = self.unnormalize_obs(obs)
        return obs

    def unnormalize_obs(self, obs):
        return obs * self._obs_iqr + self._obs_median

    def compute_rew(self, states, targets):
        signals = states[:, [self._state_fields.index(t) for t in self.target_fields]]
        return -1 * np.sum(np.abs(signals - targets), axis=1)

    def _wrap_target_dist(self, dist):
        target_median = np.array(
            [[self.info["normalization_dict"][t]["median"] for t in self.target_fields]]
        )
        target_iqr = np.array(
            [[self.info["normalization_dict"][t]["iqr"] for t in self.target_fields]]
        )

        def target_sampler(sh):
            sample = dist(sh)
            return (sample - target_median) / target_iqr

        return target_sampler

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space


class TrackingStartSelector(StartSelector):
    def __init__(self, data_path, info_dict_path, test_shots=False, shuffle=True):
        with open(info_dict_path, "rb") as f:
            self.info = pkl.load(f)
        data = load_from_hdf5(data_path)
        prefix = "te_" if test_shots else "tr_"
        curr_shot = data[prefix + "shotnums"][0]
        starts = [data[prefix + "x"][0]]
        for idx in range(1, len(data[prefix + "shotnums"])):
            if data[prefix + "shotnums"][idx] != curr_shot:
                curr_shot = data[prefix + "shotnums"][idx]
                starts.append(data[prefix + "x"][idx])
        state_idxs = [
            i for i, s in enumerate(self.info["x_columns"]) if "next_delta" not in s
        ]
        self.starts = np.vstack(starts)[:, state_idxs]
        self.shuffle = shuffle

    def get_starts(self, num_starts: int) -> np.ndarray:
        """Get start states."""
        if self.shuffle:
            indices = np.random.randint(0, len(self.starts), num_starts)
            return self.starts[indices]
        return self.starts[:num_starts]


class TrackingGymEnv(gym.Env):
    def __init__(
        self,
        model_path=DEFAULT_MODEL_PATH,
        model_info_path=DEFAULT_MODEL_INFO_PATH,
        start_data_path=DEFAULT_START_DATA_PATH,
    ):
        target_dist = lambda sh: np.tile(np.array([2, 75]), sh).reshape(sh, 2)
        self._model_env = TrackingModelEnv(
            model_path,
            model_info_path,
            target_dist,
            action_is_change=True,
            unnormalize_obs=False,
        )
        self._start_selector = TrackingStartSelector(
            start_data_path, model_info_path, shuffle=True, test_shots=False
        )
        self._state = None
        self._target = None
        self.horizon = 20
        self.periodic_dimensions = []

    def reset(self, obs=None):
        if obs is None:
            self._state = self._start_selector.get_starts(1)
        else:
            self._state = np.atleast_2d(obs)
        if self._model_env.target_distribution is not None:
            self._target = self._model_env.target_distribution(1)
        # self._target = self._model_env.target_distribution(1)
        return self._model_env.state_to_obs(self._state, self._target).flatten()

    def step(self, action):
        if not isinstance(action, np.ndarray):
            action = np.array([action])
        result = self._model_env.multi_step(self._state, action, targets=self._target)
        self._state = result["state"]
        rew = self._model_env.compute_rew(self._state, targets=self._target)
        return result["obs"].flatten(), float(rew.flatten()), False, {}

    @property
    def observation_space(self):
        return self._model_env.observation_space

    @property
    def action_space(self):
        return self._model_env.action_space


def tracking_rew(x, next_obs):
    idxes = [0, 2]
    signals = next_obs[..., idxes]
    targets = [0.4544037912481128, 0.515012974224002]
    return -1 * np.sum(np.abs(signals - targets), axis=-1)


def get_signal(field):
    if "_prev_delta" in field:
        return field[: -len("_prev_delta")]
    if "_next_delta" in field:
        return field[: -len("_next_delta")]
    return field


def test_tracking_env():
    num_trials = 50
    env = TrackingGymEnv()
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
        pred_rewards = tracking_rew(x, next_observations)
        # test reward function
        assert np.allclose(pred_rewards, rewards)
        returns.append(np.sum(rewards))
    print(f"Random actions give {np.mean(returns)} return +- {np.std(returns)}")

    # test reset to state
    for obs in observations:
        reset_obs = env.reset(obs)
        assert np.allclose(reset_obs, obs)


if __name__ == "__main__":
    test_tracking_env()
