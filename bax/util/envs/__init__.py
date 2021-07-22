from gym.envs.registration import register
from bax.util.envs.pendulum import PendulumEnv
from bax.util.envs.goddard import GoddardEnv
from bax.util.envs.pets_cartpole import PETSCartpoleEnv

# register each environment we wanna use
register(
    id='bacpendulum-v0',
    entry_point=PendulumEnv,
    )

register(
    id='goddard-v0',
    entry_point=GoddardEnv,
    )
register(
    id='petscartpole-v0',
    entry_point=PETSCartpoleEnv,
    )
