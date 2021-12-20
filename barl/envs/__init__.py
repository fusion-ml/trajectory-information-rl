import logging
from gym.envs.registration import register
from barl.envs.pendulum import PendulumEnv, pendulum_reward
from barl.envs.pilco_cartpole import (
        CartPoleSwingUpEnv,
        pilco_cartpole_reward,
        tf_pilco_cartpole_reward,
        )
from barl.envs.goddard import GoddardEnv, goddard_reward
# from barl.util.envs.pets_cartpole import PETSCartpoleEnv, cartpole_reward
from barl.envs.acrobot import AcrobotEnv, acrobot_reward
from barl.envs.lava_path import LavaPathEnv, lava_path_reward

# register each environment we wanna use
register(
    id='bacpendulum-v0',
    entry_point=PendulumEnv,
    )

register(
    id='bacpendulum-test-v0',
    entry_point=PendulumEnv,
    kwargs={'test_case': True},
    )

register(
    id='bacpendulum-tight-v0',
    entry_point=PendulumEnv,
    kwargs={'tight_start': True},
    )

register(
    id='bacpendulum-medium-v0',
    entry_point=PendulumEnv,
    kwargs={'medium_start': True},
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
register(
    id='lavapath-v0',
    entry_point=LavaPathEnv,
    )
reward_functions = {
        'bacpendulum-v0': pendulum_reward,
        'bacpendulum-test-v0': pendulum_reward,
        'bacpendulum-tight-v0': pendulum_reward,
        'bacpendulum-medium-v0': pendulum_reward,
        'goddard-v0': goddard_reward,
        # 'petscartpole-v0': cartpole_reward,
        'pilcocartpole-v0': pilco_cartpole_reward,
        'pilcocartpole-trig-v0': pilco_cartpole_reward,
        'bacrobot-v0': acrobot_reward,
        'lavapath-v0': lava_path_reward,
        }
tf_reward_functions = {
        'bacpendulum-v0': pendulum_reward,
        'pilcocartpole-v0': tf_pilco_cartpole_reward,
        }
# mujoco stuff
try:
    from barl.envs.swimmer import BACSwimmerEnv, swimmer_reward
    from barl.envs.reacher import BACReacherEnv, reacher_reward, tf_reacher_reward
    from barl.envs.half_cheetah_v3 import HalfCheetahEnv, half_cheetah_reward
    register(
        id='bacswimmer-v0',
        entry_point=BACSwimmerEnv,
        )
    register(
        id='bacswimmer-rew-v0',
        entry_point=BACSwimmerEnv,
        kwargs={'concat_reward': True}
        )
    register(
        id='bacreacher-v0',
        entry_point=BACReacherEnv,
        )
    register(
        id='bacreacher-tight-v0',
        entry_point=BACReacherEnv,
        kwargs={'tight': True},
        )
    register(
        id='bachalfcheetah-v0',
        entry_point=HalfCheetahEnv,
        )

    reward_functions['bacswimmer-v0'] = swimmer_reward
    reward_functions['bacreacher-v0'] = reacher_reward
    reward_functions['bacswimmer-rew-v0'] = swimmer_reward
    reward_functions['bacreacher-tight-v0'] = reacher_reward
    tf_reward_functions['bacreacher-v0'] = tf_reacher_reward
    reward_functions['bachalfcheetah-v0'] = half_cheetah_reward
except:
    logging.info('mujoco not found, skipping those envs')
try:
    from barl.envs.beta_tracking_env import BetaTrackingGymEnv, beta_tracking_rew
    from barl.envs.tracking_env import TrackingGymEnv, tracking_rew
    register(
        id='betatracking-v0',
        entry_point=BetaTrackingGymEnv,
        )
    register(
        id='betatracking-fixed-v0',
        entry_point=BetaTrackingGymEnv,
        kwargs={"shuffle": False},
        )
    register(
        id='plasmatracking-v0',
        entry_point=TrackingGymEnv,
        )
    reward_functions['plasmatracking-v0'] = tracking_rew
    reward_functions['betatracking-v0'] = beta_tracking_rew
    reward_functions['betatracking-fixed-v0'] = beta_tracking_rew
except:
    logging.info('fusion dependencies not found, skipping')
