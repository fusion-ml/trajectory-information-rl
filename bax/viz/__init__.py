from bax.viz.plot import plot_pendulum, plot_cartpole, plot_pilco_cartpole, plot_acrobot, noop, plot_trig_pendulum
from bax.viz.plot import plot_trig_pilco_cartpole
plotters = {
        'bacpendulum-v0': plot_pendulum,
        'bacpendulum-tight-v0': plot_pendulum,
        'bacpendulum-trig-v0': plot_trig_pendulum,
        'bacpendulum-medium-v0': plot_pendulum,
        'petscartpole-v0': plot_cartpole,
        'pilcocartpole-v0': plot_pilco_cartpole,
        'pilcocartpole-trig-v0': plot_trig_pilco_cartpole,
        'bacrobot-v0': plot_acrobot,
        'bacswimmer-v0': noop,
        'bacreacher-v0': noop,
        }
