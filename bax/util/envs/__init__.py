from gym.envs.registration import register
from bax.util.envs.pendulum import PendulumEnv, pendulum_reward
from bax.util.envs.pilco_cartpole import CartPoleSwingUpEnv, pilco_cartpole_reward
from bax.util.envs.goddard import GoddardEnv, goddard_reward
# from bax.util.envs.pets_cartpole import PETSCartpoleEnv, cartpole_reward
from bax.util.envs.acrobot import AcrobotEnv, acrobot_reward

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
    id='bacpendulum-medium-v0',
    entry_point=PendulumEnv,
    kwargs={'medium_start': True}
    )

register(
    id='goddard-v0',
    entry_point=GoddardEnv,
    )
# register(
#     id='petscartpole-v0',
#     entry_point=PETSCartpoleEnv,
#     )
register(
    id='pilcocartpole-v0',
    entry_point=CartPoleSwingUpEnv,
    )
register(
    id='pilcocartpole-trig-v0',
    entry_point=CartPoleSwingUpEnv,
    kwargs={'use_trig': True},
    )
register(
    id='bacrobot-v0',
    entry_point=AcrobotEnv,
    )
reward_functions = {
        'bacpendulum-v0': pendulum_reward,
        'bacpendulum-tight-v0': pendulum_reward,
        'bacpendulum-medium-v0': pendulum_reward,
        'goddard-v0': goddard_reward,
        # 'petscartpole-v0': cartpole_reward,
        'pilcocartpole-v0': pilco_cartpole_reward,
        'pilcocartpole-trig-v0': pilco_cartpole_reward,
        'bacrobot-v0': acrobot_reward,
        }
