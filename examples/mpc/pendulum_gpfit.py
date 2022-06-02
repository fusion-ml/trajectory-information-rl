"""
Show GP hyperparameter fitting on pendulum example.
"""

from argparse import Namespace
import numpy as np

from bax.models.stan_gp import get_stangp_hypers_from_data
from bax.alg.mpc import MPC
from bax.util.envs.pendulum import PendulumEnv, pendulum_reward
from bax.util.control_util import get_f_mpc
from bax.util.domain_util import unif_random_sample_domain


# Set random seed
seed = 11
np.random.seed(seed)


# -------------
# Start Script
# -------------
# Set black-box function
env = PendulumEnv(seed=seed)
obs_dim = env.observation_space.low.size
action_dim = env.action_space.low.size

plan_env = PendulumEnv(seed=seed)
f = get_f_mpc(plan_env)
start_obs = env.reset()

# Set domain
low = np.concatenate([env.observation_space.low, env.action_space.low])
high = np.concatenate([env.observation_space.high, env.action_space.high])
domain = [elt for elt in zip(low, high)]

# Set algorithm
algo_class = MPC
algo_params = dict(
    start_obs=start_obs,
    env=plan_env,
    reward_function=pendulum_reward,
    base_nsamps=10,
    planning_horizon=20,
    n_elites=3,
    beta=3,
    gamma=1.25,
    xi=0.3,
    num_iters=3,
    actions_per_plan=6,
    domain=domain,
)
algo = algo_class(algo_params)

# Set data
data = Namespace()
n_init_data = 3000
data.x = unif_random_sample_domain(domain, n_init_data)
data.y = [f(xi) for xi in data.x]

# Fit hypers
data_fit_0 = Namespace(x=data.x, y=[yi[0] for yi in data.y])
data_fit_1 = Namespace(x=data.x, y=[yi[1] for yi in data.y])
gp_params_0 = get_stangp_hypers_from_data(data_fit_0)
gp_params_1 = get_stangp_hypers_from_data(data_fit_1)
