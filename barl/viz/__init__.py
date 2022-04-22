from collections import defaultdict
from barl.viz.plot import (
    plot_pendulum,
    plot_cartpole,
    plot_pilco_cartpole,
    plot_acrobot,
    # noop,
    plot_lava_path,
    plot_weird_gain,
    make_plot_obs,
    plot_generic,
    scatter,
    plot,
    noop,
)

_plotters = {
    "bacpendulum-v0": plot_pendulum,
    "bacpendulum-test-v0": plot_pendulum,
    "bacpendulum-tight-v0": plot_pendulum,
    "bacpendulum-medium-v0": plot_pendulum,
    "petscartpole-v0": plot_cartpole,
    "pilcocartpole-v0": plot_pilco_cartpole,
    "bacrobot-v0": plot_acrobot,
    "bacswimmer-v0": plot_generic,
    "bacreacher-v0": plot_generic,
    "bacreacher-tight-v0": plot_generic,
    "lavapath-v0": plot_lava_path,
    "shortlavapath-v0": plot_lava_path,
    "betatracking-v0": plot_generic,
    "betatracking-fixed-v0": plot_generic,
    "plasmatracking-v0": plot_generic,
    "bachalfcheetah-v0": noop,
    "weirdgain-v0": plot_weird_gain,
}
plotters = defaultdict(lambda: plot_generic)
plotters.update(_plotters)
