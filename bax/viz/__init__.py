from bax.viz.plot import plot_pendulum, plot_cartpole, plot_pilco_cartpole, plot_acrobot
plotters = {
        'bacpendulum-v0': plot_pendulum,
        'bacpendulum-tight-v0': plot_pendulum,
        'bacpendulum-medium-v0': plot_pendulum,
        'petscartpole-v0': plot_cartpole,
        'pilcocartpole-v0': plot_pilco_cartpole,
        'bacrobot-v0': plot_acrobot,
        }
