from matplotlib import pyplot as plt

def plot_pendulum(path, ax=None, domain=None, path_str="samp"):
    """Plot a path through an assumed two-dimensional state space."""
    assert path_str in ["samp", "true", "postmean"]
    if ax is None:
        assert domain is not None
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        ax.set(
            xlim=(domain[0][0], domain[0][1]),
            ylim=(domain[1][0], domain[1][1]),
            xlabel='$\\theta$',
            ylabel='$\\dot{\\theta}$',
        )


    x_plot = [xi[0] for xi in path.x]
    y_plot = [xi[1] for xi in path.x]

    if path_str == "true":
        ax.plot(x_plot, y_plot, 'k--', linewidth=3)
        ax.plot(x_plot, y_plot, '*', color='k', markersize=5)
    elif path_str == "postmean":
        ax.plot(x_plot, y_plot, 'r--', linewidth=3)
        ax.plot(x_plot, y_plot, '*', color='r', markersize=5)
    elif path_str == "samp":
        ax.plot(x_plot, y_plot, 'k--', linewidth=1, alpha=0.3)
        ax.plot(x_plot, y_plot, 'o', alpha=0.3)
    return ax


def plot_cartpole(path, ax=None, domain=None, path_str="samp"):
    """Plot a path through an assumed two-dimensional state space."""
    assert path_str in ["samp", "true", "postmean"]
    if ax is None:
        assert domain is not None
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        ax.set(
            xlim=(domain[0][0], domain[0][1]),
            ylim=(domain[1][0], domain[1][1]),
            xlabel='$x$',
            ylabel='$\\theta$',
        )


    x_plot = [xi[0] for xi in path.x]
    y_plot = [xi[1] for xi in path.x]

    if path_str == "true":
        ax.plot(x_plot, y_plot, 'k--', linewidth=3)
        ax.plot(x_plot, y_plot, '*', color='k', markersize=5)
    elif path_str == "postmean":
        ax.plot(x_plot, y_plot, 'r--', linewidth=3)
        ax.plot(x_plot, y_plot, '*', color='r', markersize=5)
    elif path_str == "samp":
        ax.plot(x_plot, y_plot, 'k--', linewidth=1, alpha=0.3)
        ax.plot(x_plot, y_plot, 'o', alpha=0.3)
    return ax

def plot_pilco_cartpole(path, ax=None, domain=None, path_str="samp"):
    """Plot a path through an assumed two-dimensional state space."""
    assert path_str in ["samp", "true", "postmean"]
    if ax is None:
        assert domain is not None
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        ax.set(
            xlim=(domain[0][0], domain[0][1]),
            ylim=(domain[1][0], domain[1][1]),
            xlabel='$x$',
            ylabel='$\\theta$',
        )


    x_plot = [xi[0] for xi in path.x]
    y_plot = [xi[2] for xi in path.x]

    if path_str == "true":
        ax.plot(x_plot, y_plot, 'k--', linewidth=3)
        ax.plot(x_plot, y_plot, '*', color='k', markersize=5)
    elif path_str == "postmean":
        ax.plot(x_plot, y_plot, 'r--', linewidth=3)
        ax.plot(x_plot, y_plot, '*', color='r', markersize=5)
    elif path_str == "samp":
        ax.plot(x_plot, y_plot, 'k--', linewidth=1, alpha=0.3)
        ax.plot(x_plot, y_plot, 'o', alpha=0.3)
    return ax
