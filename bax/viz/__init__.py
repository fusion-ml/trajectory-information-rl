from bax.viz.plot import (
    plot_pendulum,
    plot_cartpole,
    plot_pilco_cartpole,
    plot_acrobot,
    noop,
    plot_lava_path,
    make_plot_obs,
    )
plotters = {
        'bacpendulum-v0': plot_pendulum,
        'bacpendulum-tight-v0': plot_pendulum,
        'bacpendulum-medium-v0': plot_pendulum,
        'petscartpole-v0': plot_cartpole,
        'pilcocartpole-v0': plot_pilco_cartpole,
        'bacrobot-v0': plot_acrobot,
        'bacswimmer-v0': noop,
        'bacreacher-v0': noop,
        'bacreacher-tight-v0': noop,
        'lavapath-v0': plot_lava_path,
        'betatracking-v0': noop,
        'plasmatracking-v0': noop,
        }
