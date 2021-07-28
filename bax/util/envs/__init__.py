from gym.envs.registration import register
from bax.util.envs.pendulum import PendulumEnv, pendulum_reward
from bax.util.envs.goddard import GoddardEnv, goddard_reward
from bax.util.envs.pets_cartpole import PETSCartpoleEnv

# register each environment we wanna use
register(
    id='bacpendulum-v0',
    entry_point=PendulumEnv,
    )

register(
    id='bacpendulum-tight-v0',
    entry_point=PendulumEnv,
    kwargs={'tight_start': True}
    )

register(
    id='goddard-v0',
    entry_point=GoddardEnv,
    )
register(
    id='petscartpole-v0',
    entry_point=PETSCartpoleEnv,
    )
reward_functions = {
        'bacpendulum-v0': pendulum_reward,
        'bacpendulum-tight-v0': pendulum_reward,
        'goddard-v0': goddard_reward,
        }
