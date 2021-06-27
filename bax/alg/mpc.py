"""
Evolution strategies as algorithms for BAX.
"""

from argparse import Namespace
import copy
import numpy as np
from math import ceil


from .algorithms import Algorithm
from ..util.misc_util import dict_to_namespace
from ..util.control_util import compute_return, iCEM_generate_samples
from ..util.domain_util import project_to_domain


class MPC(Algorithm):
    """
    An algorithm for model-predictive control. Here, the queries are concatenated states and actions and the output of the query is the next state.
    We need the reward function in our algorithm as well as a start state.
    """

    def set_params(self, params):
        """Set self.params, the parameters for the algorithm."""
        super().set_params(params)
        params = dict_to_namespace(params)

        self.params.name = getattr(params, "name", "MPC")
        self.params.start_obs = params.start_obs
        self.params.env = params.env
        self.params.discount_factor = getattr(params, 'discount_factor', 0.99)
        # reward function is currently required, needs to take (state x action) x next_obs -> R
        self.params.reward_function = params.reward_function
        self.terminal_function = self.params.terminal_function = getattr(params, "terminal_function", None)
        self.params.env_horizon = params.env.horizon
        self.params.action_dim = params.env.action_space.low.size
        self.params.obs_dim = params.env.observation_space.low.size
        self.params.action_lower_bound = getattr(params, "action_lower_bound", -1)
        self.params.action_upper_bound = getattr(params, "action_upper_bound", 1)
        self.params.initial_variance_divisor = getattr(params, "initial_variance_divisor", 4)
        self.params.base_nsamps = getattr(params, "base_nsamps", 8)
        self.params.planning_horizon = getattr(params, "planning_horizon", 10)
        self.params.n_elites = getattr(params, "n_elites", 4)
        self.params.beta = getattr(params, "beta", 3)
        self.params.gamma = getattr(params, "gamma", 1.25)
        self.params.xi = getattr(params, "xi", 0.3)
        self.params.num_iters = getattr(params, "num_iters", 3)
        self.params.actions_per_plan = getattr(params, "actions_per_plan", 4)
        self.params.project_to_domain = getattr(params, 'project_to_domain', False)
        self.params.domain = getattr(params, "domain", None)
        self.traj_samples = None
        self.traj_states = None
        self.traj_rewards = None
        self.current_traj_idx = None
        self.current_t = None
        self.shifted_actions = []
        self.shifted_states = []
        self.shifted_rewards = []
        self.planned_states = []
        self.planned_actions = []
        self.planned_rewards = []
        self.saved_states = []
        self.saved_actions = []
        self.saved_rewards = []
        self.mean = None
        self.var = None
        self.iter_num = None
        self.shift_done = True
        self.samples_done = False
        self.current_obs = None
        self.best_return = -np.inf
        self.best_actions = None
        self.best_obs = None
        self.best_rewards = None

    def initialize(self):
        """Initialize algorithm, reset execution path."""
        super().initialize()

        # set up initial CEM distribution
        self.mean = np.zeros((self.params.planning_horizon, self.params.action_dim))
        self.var = np.ones_like(self.mean) * ((self.params.action_upper_bound - self.params.action_lower_bound) /
                                                 self.params.initial_variance_divisor) ** 2
        initial_nsamps = int(max(self.params.base_nsamps * (self.params.gamma ** -1), 2 * self.params.n_elites))
        self.traj_samples = iCEM_generate_samples(initial_nsamps,
                                                  self.params.planning_horizon,
                                                  self.params.beta,
                                                  self.mean,
                                                  self.var,
                                                  self.params.action_lower_bound,
                                                  self.params.action_upper_bound)
        self.traj_samples = list(self.traj_samples)
        self.current_traj_idx = 0
        # this one is for CEM
        self.current_t_plan = 0
        # this one is for the actual agent
        self.current_t = 0
        self.current_obs = self.params.start_obs
        self.iter_num = 0
        self.shift_done = True
        self.samples_done = False
        self.planned_states = [self.params.start_obs]
        self.planned_actions = []
        self.planned_rewards = []
        self.saved_states = []
        self.saved_actions = []
        self.saved_rewards = []
        self.shifted_actions = []
        self.shifted_states = []
        self.shifted_rewards = []
        self.traj_states = [[] for _ in range(initial_nsamps)]
        self.traj_rewards = [[] for _ in range(initial_nsamps)]
        self.best_return = -np.inf
        self.best_actions = None
        self.best_obs = None
        self.best_rewards = None

    def get_next_x(self):
        """
        Given the current execution path, return the next x in the execution path. If
        the algorithm is complete, return None.
        """
        if len(self.exe_path.x) > 0:
            self.process_prev_output()
        # at this point the *_done should be correct
        if self.samples_done and self.iter_num + 1 == self.params.num_iters:
            self.save_planned_actions()
            if self.current_t + 1 >= self.params.env_horizon:
                # done planning
                return None
            self.reset_CEM()
        elif self.samples_done:
            self.resample_CEM()
        if not self.shift_done:
            # do all the shifted ones first
            query = self.get_shift_x()
        elif not self.samples_done:
            query = self.get_sample_x()

        # Optionally, project to domain
        if self.params.project_to_domain:
            query = project_to_domain(query, self.params.domain)

        return query

    def get_shift_x(self):
        if len(self.shifted_states[self.current_traj_idx]) > 0:
            obs = self.shifted_states[self.current_traj_idx][-1]
        else:
            obs = self.current_obs
        action = self.shifted_actions[self.current_traj_idx][self.current_t_plan]
        query = np.concatenate([obs, action])
        return query

    def get_sample_x(self):
        if len(self.traj_states[self.current_traj_idx]) > 0:
            obs = self.traj_states[self.current_traj_idx][-1]
        else:
            obs = self.current_obs
        action = self.traj_samples[self.current_traj_idx][self.current_t_plan]
        query = np.concatenate([obs, action])
        return query

    def resample_CEM(self):
        self.iter_num += 1
        nsamps = int(max(self.params.base_nsamps * (self.params.gamma ** -self.iter_num), 2 * self.params.n_elites))
        all_rewards = self.traj_rewards + self.shifted_rewards + self.saved_rewards
        all_states = self.traj_states + self.shifted_states + self.saved_states
        all_actions = self.traj_samples + self.shifted_actions + self.saved_actions
        all_returns = [compute_return(rewards, self.params.discount_factor) for rewards in all_rewards]
        best_idx = np.argmax(all_returns)
        best_current_return = all_returns[best_idx]
        if best_current_return > self.best_return:
            self.best_return = best_current_return
            self.best_actions = all_actions[best_idx]
            self.best_obs = all_states[best_idx]
            self.best_rewards = all_rewards[best_idx]
        elite_idx = np.argsort(all_returns)[-self.params.n_elites:]
        elites = np.array(all_actions)[elite_idx, :]
        mean = np.mean(elites, axis=0)
        var = np.var(elites, axis=0)
        samples = iCEM_generate_samples(nsamps, self.params.planning_horizon, self.params.beta, mean, var,
                                        self.params.action_lower_bound, self.params.action_upper_bound)
        n_save_elites = ceil(self.params.n_elites * self.params.xi)
        save_idx = elite_idx[-n_save_elites:]
        self.saved_actions = [all_actions[idx] for idx in save_idx]
        self.saved_states = [all_states[idx] for idx in save_idx]
        self.saved_rewards = [all_rewards[idx] for idx in save_idx]
        if self.iter_num + 1 == self.params.num_iters:
            samples = np.concatenate([samples, mean[None, :]], axis=0)
        self.traj_samples = list(samples)
        self.traj_states = [[] for _ in range(len(self.traj_samples))]
        self.traj_rewards = [[] for _ in range(len(self.traj_samples))]
        self.shifted_states = []
        self.shifted_actions = []
        self.shifted_rewards = []
        self.current_traj_idx = 0
        self.samples_done = False

    def get_next_obs(self, x, y):
        start_obs = x[:self.params.obs_dim]
        delta_obs = y
        return start_obs + delta_obs

    def process_prev_output(self):
        next_obs = self.get_next_obs(self.exe_path.x[-1], self.exe_path.y[-1])
        reward = self.params.reward_function(self.exe_path.x[-1], next_obs)
        if not self.shift_done:
            # do all the shift stuff
            self.shifted_states[self.current_traj_idx].append(next_obs)
            self.shifted_rewards[self.current_traj_idx].append(reward)
            self.current_t_plan += 1
            if self.current_t_plan == self.params.planning_horizon:
                self.current_t_plan = 0
                self.current_traj_idx += 1
            if self.current_traj_idx == len(self.shifted_states):
                self.current_traj_idx = 0
                self.current_t_plan = 0
                self.shift_done = True
            return
        # otherwise do the stuff for the standard CEM
        self.traj_states[self.current_traj_idx].append(next_obs)
        self.traj_rewards[self.current_traj_idx].append(reward)
        self.current_t_plan += 1
        if self.current_t_plan == self.params.planning_horizon:
            self.current_t_plan = 0
            self.current_traj_idx += 1
        if self.current_traj_idx == len(self.traj_samples):
            self.samples_done = True

    def save_planned_actions(self):
        # after CEM is complete for the current timestep, "execute" the best actions
        # and adjust the time and current state accordingly
        all_rewards = self.traj_rewards + self.shifted_rewards + self.saved_rewards + [self.best_rewards]
        all_states = self.traj_states + self.shifted_states + self.saved_states + [self.best_obs]
        all_actions = self.traj_samples + self.shifted_actions + self.saved_actions + [self.best_actions]
        all_returns = [compute_return(rewards, self.params.discount_factor) for rewards in all_rewards]
        best_sample_idx = np.argmax(all_returns)
        best_actions = all_actions[best_sample_idx]
        best_states = all_states[best_sample_idx]
        best_rewards = all_rewards[best_sample_idx]
        for t in range(self.params.actions_per_plan):
            self.planned_actions.append(best_actions[t])
            self.planned_states.append(best_states[t])
            self.planned_rewards.append(best_rewards[t])
        self.current_t += self.params.actions_per_plan
        # since we don't have the start state in this list, we have to subtract 1 here
        # this should be where the current plan leaves us
        self.current_obs = best_states[self.params.actions_per_plan - 1]
        self.shift_samples(all_returns, all_states, all_actions, all_rewards)

    def reset_CEM(self):
        self.mean = np.concatenate([self.mean[self.params.actions_per_plan:],
                                    np.zeros((self.params.actions_per_plan, self.params.action_dim))])
        self.var = np.ones_like(self.mean) * ((self.params.action_upper_bound - self.params.action_lower_bound) /
                                                 self.params.initial_variance_divisor) ** 2
        self.iter_num = 0
        initial_nsamps = int(max(self.params.base_nsamps * (self.params.gamma ** -1), 2 * self.params.n_elites))
        self.traj_samples = iCEM_generate_samples(initial_nsamps,
                                                  self.params.planning_horizon,
                                                  self.params.beta,
                                                  self.mean,
                                                  self.var,
                                                  self.params.action_lower_bound,
                                                  self.params.action_upper_bound)
        self.traj_samples = list(self.traj_samples)
        self.traj_states = [[] for _ in range(initial_nsamps)]
        self.traj_rewards = [[] for _ in range(initial_nsamps)]
        self.saved_actions = []
        self.saved_states = []
        self.saved_rewards = []
        self.samples_done = False
        self.current_traj_idx = 0
        self.best_return = -np.inf
        self.best_actions = None
        self.best_obs = None
        self.best_rewards = None

    def shift_samples(self, all_returns, all_states, all_actions, all_rewards):
        n_keep = ceil(self.params.xi * self.params.n_elites)
        keep_indices = np.argsort(all_returns)[-n_keep:]
        self.shifted_states = []
        self.shifted_actions = []
        self.shifted_rewards = []
        for idx in keep_indices:
            new_actions = np.array([self.params.env.action_space.sample() for _ in range(self.params.actions_per_plan)])
            self.shifted_actions.append(np.concatenate([all_actions[idx][self.params.actions_per_plan:], new_actions]))
            self.shifted_rewards.append([])
            self.shifted_states.append([])
        self.current_t_plan = 0
        self.shift_done = False

    def get_output(self):
        """Given an execution path, return algorithm output."""
        return self.planned_states, self.planned_actions, self.planned_rewards

    def get_exe_path_crop(self):
        """
        Return the minimal execution path for output, i.e. cropped execution path,
        specific to this algorithm.
        """
        exe_path_crop = Namespace(x=[], y=[])
        for i, (obs, action) in enumerate(zip(self.planned_states, self.planned_actions)):
            next_obs = self.planned_states[i + 1]
            x = np.concatenate([obs, action])

            # Optionally, project to domain
            if self.params.project_to_domain:
                x = project_to_domain(x, self.params.domain)

            y = next_obs - obs
            exe_path_crop.x.append(x)
            exe_path_crop.y.append(y)
            if self.terminal_function is not None and self.terminal_function(x, next_obs):
                break
        return exe_path_crop


def test_MPC_algorithm():
    from util.envs.continuous_cartpole import ContinuousCartPoleEnv, continuous_cartpole_reward
    from util.control_util import ResettableEnv, get_f_mpc
    env = ContinuousCartPoleEnv()
    obs_dim = env.observation_space.low.size
    action_dim = env.action_space.low.size
    plan_env = ResettableEnv(ContinuousCartPoleEnv())
    f = get_f_mpc(plan_env)
    start_obs = env.reset()
    params = dict(
            start_obs=start_obs,
            env=plan_env,
            reward_function=continuous_cartpole_reward)
    mpc = MPC(params)
    mpc.initialize()
    path, output = mpc.run_algorithm_on_f(f)
    observations, actions, rewards = output
    total_return = sum(rewards)
    print(f"MPC gets {total_return} return with {len(path.x)} queries based on itself")
    done = False
    rewards = []
    for i, action in enumerate(actions):
        next_obs, rew, done, info = env.step(action)
        if (next_obs != observations[i + 1]).any():
            error = np.linalg.norm(next_obs - observations[i + 1])
            print(f"i={i}, error={error}")
        rewards.append(rew)
        if done:
            break
    real_return = compute_return(rewards, 1.)
    print(f"based on the env it gets {real_return} return")



if __name__ == '__main__':
    test_MPC_algorithm()
