import gym
from gym.spaces import Box
import numpy as np

# import cv2
# import matplotlib.pyplot as plt
# import matplotlib.patches as patches
# from matplotlib.backends.backend_agg import FigureCanvas


def in_lava(x, lava_pits):
    # TODO add intersection checking to make sure we don't jump over lava

    lava = False
    for lava_pit in lava_pits:
        lava = lava_pit.contains(x) | lava

    return lava


class LavaPathEnv(gym.Env):

    lava_penalty = -500
    goal = np.array([0, 10])
    periodic_dimensions = []

    # For each lava pit, we have
    lava_pits = [
        Box(low=np.array([-10, -8]), high=np.array([-0.5, 8])),
        Box(low=np.array([0.5, -8]), high=np.array([10, 8])),
    ]

    def __init__(self):
        # Observation space is [x, y, x_dot, y_dot]
        self.observation_space = Box(
            low=np.array([-20, -20, -10, -10]), high=np.array([20, 20, 10, 10])
        )

        # Action space is [F_x, F_y]
        self.action_space = Box(low=np.array([-1, -1]), high=np.array([1, 1]))

        # Simulation time step - we assume force is constant during time step
        self.dt = 0.1

        # Mass of object in kg
        self.mass = 1

        # Goal position

        self.enable_lava_walls = False

        # Region where our agent can move, agent cannot leave this area
        self.playable_area = Box(low=np.array([-20, -20]), high=np.array([20, 20]))

        # Minimum distance in meters from goal to be considered at_goal
        self.goal_delta = 0.5

        # self.timeout_steps = 200
        self.x = None
        self.x_dot = None
        self.horizon = 200

    def construct_obs(self):
        return np.concatenate([self.x, self.x_dot])

    def did_intersect(self, start, end, edge):
        return start <= edge < end

    def check_lava_wall_collision(self, new_x, new_x_dot):
        for lava_pit in self.lava_pits:

            # Check if we crossed bottom edge in this step
            if (
                self.did_intersect(self.x[0], new_x[0], lava_pit.low[0])
                and lava_pit.low[1] <= new_x[1] <= lava_pit.high[1]
            ):
                new_x[0] = lava_pit.low[0]
                new_x_dot[0] = 0

                return new_x, new_x_dot

            # Check if we crossed top edge point in this step
            if (
                self.did_intersect(self.x[0], new_x[0], lava_pit.high[0])
                and lava_pit.low[1] <= new_x[1] <= lava_pit.high[1]
            ):
                new_x[0] = lava_pit.high[0]
                new_x_dot[0] = 0

                return new_x, new_x_dot

            # Check if we crossed left edge in this step
            if (
                self.did_intersect(self.x[1], new_x[1], lava_pit.low[1])
                and lava_pit.low[0] <= new_x[0] <= lava_pit.high[0]
            ):
                new_x[1] = lava_pit.low[1]
                new_x_dot[1] = 0

                return new_x, new_x_dot

            # Check if we crossed right edge point in this step
            if (
                self.did_intersect(self.x[1], new_x[1], lava_pit.high[1])
                and lava_pit.low[0] <= new_x[0] <= lava_pit.high[0]
            ):
                new_x[1] = lava_pit.high[1]
                new_x_dot[1] = 0

                return new_x, new_x_dot

        return new_x, new_x_dot

    def at_goal(self):
        return np.sum((self.x - self.goal) ** 2) < self.goal_delta

    def get_matplotlib_image(self):
        # fig = plt.figure()
        canvas = FigureCanvas(fig)
        ax = fig.subplots()

        ax.scatter(self.x[0], self.x[1], color="blue")
        ax.scatter(self.goal[0], self.goal[1], color="green")

        # Draw left rectangle
        for lava_pit in self.lava_pits:
            delta = lava_pit.high - lava_pit.low
            patch = patches.Rectangle(
                lava_pit.low, delta[0], delta[1], fill=True, color="red"
            )

            ax.add_patch(patch)

        ax.set_xlim(self.observation_space.low[0], self.observation_space.high[0])
        ax.set_ylim(self.observation_space.low[1], self.observation_space.high[1])

        canvas.draw()
        img = np.array(canvas.renderer.buffer_rgba())
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # plt.close(fig)

        return img

    def reset(self, obs=None):
        if obs is None:
            # Start at the bottom of map, centered w.r.t bridge
            self.x = np.array([0, -10]) + np.random.uniform([-0.5, -0.5], [0.5, 0.5])

            # Start with zero velocity
            self.x_dot = np.array([0, 0])

        else:
            self.x = obs[:2]
            self.x_dot = obs[2:]

        # self.steps = 0

        return self.construct_obs()

    def step(self, action):
        # Action is [F_x, F_y]

        # Get acceleration
        a = action / self.mass

        new_x_dot = a * self.dt + self.x_dot
        new_x = 0.5 * a * self.dt**2 + self.x_dot * self.dt + self.x

        # Bound checking
        # If we're west of the playable area
        if new_x[0] < self.playable_area.low[0]:
            new_x[0] = self.playable_area.low[0]
            new_x_dot[0] = 0
        # If we're east of the playable area
        elif new_x[0] > self.playable_area.high[0]:
            new_x[0] = self.playable_area.high[0]
            new_x_dot[0] = 0

        # If we're south of the playable area
        if new_x[1] < self.playable_area.low[1]:
            new_x[1] = self.playable_area.low[1]
            new_x_dot[1] = 0
        # If we're right of the playable area
        elif new_x[1] > self.playable_area.high[1]:
            new_x[1] = self.playable_area.high[1]
            new_x_dot[1] = 0

        # Lava wall bound checking
        if self.enable_lava_walls:
            new_x, new_x_dot = self.check_lava_wall_collision(new_x, new_x_dot)

        # Update state aftewards
        self.x = new_x
        self.x_dot = new_x_dot

        # Compute done conditions
        lava = in_lava(self.x, self.lava_pits) and not self.enable_lava_walls
        # goal = self.at_goal()

        # self.steps += 1

        done = False  # lava or goal or (self.steps > self.timeout_steps)

        # Reward at each step is the negative squared distance between the current position and the goal
        # Add negative penalty
        reward = -np.sum((self.x - self.goal) ** 2) / 800 + self.lava_penalty * int(
            lava
        )

        info = {}

        return self.construct_obs(), reward, done, info

    def render(self):
        return self.get_matplotlib_image()


class ShortLavaPathEnv(LavaPathEnv):
    def __init__(self):
        super().__init__()
        self.horizon = 20

    def step(self, action):
        # we ignore the info from the first step (this allows for some "clipping"
        # of the lava, but we already allowed it a little less)
        o1, r1, d1, i1 = super().step(action)
        o2, r2, d2, i2 = super().step(action)
        return o2, r2, d2, i2


def lava_path_reward(x, next_obs):
    x_prob = next_obs[..., :2]
    if x.ndim == 1:
        lava = in_lava(x_prob, LavaPathEnv.lava_pits)
        reward = -np.sum(
            (x_prob - LavaPathEnv.goal) ** 2
        ) / 800 + LavaPathEnv.lava_penalty * int(lava)
    else:
        lava = np.array([in_lava(xi, LavaPathEnv.lava_pits) for xi in x_prob]).astype(
            int
        )
        reward = (
            -np.sum((x_prob - LavaPathEnv.goal) ** 2, axis=-1) / 800
            + LavaPathEnv.lava_penalty * lava
        )
    return reward


def test_lava_path():
    env = LavaPathEnv()
    n_tests = 100
    observations = []
    actions = []
    next_observations = []
    rewards = []
    for _ in range(n_tests):
        obs = env.reset()
        observations.append(obs)
        action = env.action_space.sample()
        actions.append(action)
        next_obs, rew, done, info = env.step(action)
        next_observations.append(next_obs)
        rewards.append(rew)
        x = np.concatenate([obs, action])
        other_rew = lava_path_reward(x, next_obs)
        assert np.allclose(rew, other_rew)
        new_obs = env.reset(obs)
        assert np.allclose(new_obs, obs)
    observations = np.array(observations)
    actions = np.array(actions)
    next_observations = np.array(next_observations)
    rewards = np.array(rewards)
    x = np.concatenate([observations, actions], axis=1)
    pred_rewards = lava_path_reward(x, next_observations)
    assert np.allclose(pred_rewards, rewards)
    done = False
    env.reset()
    for _ in range(env.horizon):
        action = env.action_space.sample()
        n, r, done, info = env.step(action)
        if done:
            break
    print("passed")


if __name__ == "__main__":
    test_lava_path()
