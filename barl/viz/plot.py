from matplotlib import pyplot as plt
from math import ceil
import numpy as np
from copy import deepcopy
from barl.envs.pilco_cartpole import get_pole_pos
import matplotlib.patches as patches
from barl.envs.lava_path import LavaPathEnv
from barl.envs.weird_gain import GOAL as WEIRD_GAIN_GOAL


def plot_generic(path, ax=None, fig=None, domain=None, path_str="samp", env=None):
    assert path_str in ["samp", "true", "postmean"]
    if ax is None:
        if path:
            ndimx = len(path.x[0])
            nplots = int(ceil(ndimx / 2))
        elif domain:
            ndimx = len(domain)
            nplots = int(ceil(ndimx / 2))
        fig, axes = plt.subplots(1, nplots, figsize=(5 * nplots, 5))
        if domain:
            domain = deepcopy(domain)
            if len(domain) % 2 == 1:
                domain.append([-1, 1])
            for i, ax in enumerate(axes):
                ax.set(
                    xlim=(domain[i * 2][0], domain[i * 2][1]),
                    ylim=(domain[i * 2 + 1][0], domain[i * 2 + 1][1]),
                    xlabel=f"$x_{i * 2}$",
                    ylabel=f"$x_{i * 2 + 1}$",
                )
        if path is None:
            return axes, fig
    else:
        axes = ax
    for i, ax in enumerate(axes):
        x_plot = [xi[2 * i] for xi in path.x]
        try:
            y_plot = [xi[2 * i + 1] for xi in path.x]
        except IndexError:

            y_plot = [0] * len(path.x)
        if path_str == "true":
            ax.plot(x_plot, y_plot, "k--", linewidth=3)
            ax.plot(x_plot, y_plot, "*", color="k", markersize=5)
        elif path_str == "postmean":
            ax.plot(x_plot, y_plot, "r--", linewidth=3)
            ax.plot(x_plot, y_plot, "*", color="r", markersize=5)
        elif path_str == "samp":
            lines2d = ax.plot(x_plot, y_plot, "--", linewidth=1, alpha=0.3)
            ax.plot(x_plot, y_plot, "o", color=lines2d[0].get_color(), alpha=0.3)

        # Also plot small indicator of start-of-path
        ax.plot(x_plot[0], y_plot[0], "<", markersize=2, color="k", alpha=0.5)

    return axes, fig


def plot_pendulum(path, ax=None, fig=None, domain=None, path_str="samp", env=None):
    """Plot a path through an assumed two-dimensional state space."""
    assert path_str in ["samp", "true", "postmean"]
    if ax is None:
        assert domain is not None
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        ax.set(
            xlim=(domain[0][0], domain[0][1]),
            ylim=(domain[1][0], domain[1][1]),
            xlabel="$\\theta$",
            ylabel="$\\dot{\\theta}$",
        )
        if path is None:
            return ax, fig
    x_plot = [xi[0] for xi in path.x]
    y_plot = [xi[1] for xi in path.x]

    if path_str == "true":
        ax.plot(x_plot, y_plot, "k--", linewidth=3)
        ax.plot(x_plot, y_plot, "*", color="k", markersize=5)
    elif path_str == "postmean":
        ax.plot(x_plot, y_plot, "r--", linewidth=3)
        ax.plot(x_plot, y_plot, "*", color="r", markersize=5)
    # elif path_str == "samp":
    # ax.plot(x_plot, y_plot, 'k--', linewidth=1, alpha=0.3)
    # ax.plot(x_plot, y_plot, 'o', alpha=0.3)
    elif path_str == "samp":
        lines2d = ax.plot(x_plot, y_plot, "--", linewidth=1, alpha=0.3)
        ax.plot(x_plot, y_plot, "o", color=lines2d[0].get_color(), alpha=0.3)

    # Also plot small indicator of start-of-path
    ax.plot(x_plot[0], y_plot[0], "<", markersize=2, color="k", alpha=0.5)

    return ax, fig


def plot_lava_path(path, ax=None, fig=None, domain=None, path_str="samp", env=None):
    """Plot a path through an assumed two-dimensional state space."""
    assert path_str in ["samp", "true", "postmean"], f"path_str is {path_str}"
    if ax is None:
        assert domain is not None
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        ax.set(
            xlim=(domain[0][0], domain[0][1]),
            ylim=(domain[1][0], domain[1][1]),
            xlabel="$x$",
            ylabel="$y$",
        )

    # Draw left rectangle
    for lava_pit in LavaPathEnv.lava_pits:
        delta = lava_pit.high - lava_pit.low
        patch = patches.Rectangle(
            lava_pit.low, delta[0], delta[1], fill=True, color="orange", zorder=-100
        )

        ax.add_patch(patch)
    if path is None:
        return ax, fig

    x_plot = [xi[0] for xi in path.x]
    y_plot = [xi[1] for xi in path.x]

    if path_str == "true":
        ax.plot(x_plot, y_plot, "k--", linewidth=3)
        ax.plot(x_plot, y_plot, "*", color="k", markersize=5)
    elif path_str == "postmean":
        ax.plot(x_plot, y_plot, "r--", linewidth=3)
        ax.plot(x_plot, y_plot, "*", color="r", markersize=5)
    elif path_str == "samp":
        ax.plot(x_plot, y_plot, "k--", linewidth=1, alpha=0.3)
        ax.plot(x_plot, y_plot, "o", alpha=0.3)
    ax.scatter(
        LavaPathEnv.goal[0], LavaPathEnv.goal[1], color="green", s=100, zorder=99
    )
    return ax, fig


def plot_weird_gain(path, ax=None, fig=None, domain=None, path_str="samp", env=None):
    """Plot a path through an assumed two-dimensional state space."""
    assert path_str in ["samp", "true", "postmean"], f"path_str is {path_str}"
    if ax is None:
        assert domain is not None
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        ax.set(
            xlim=(domain[0][0], domain[0][1]),
            ylim=(domain[1][0], domain[1][1]),
            xlabel="$x$",
            ylabel="$y$",
        )

    if path is None:
        return ax, fig

    x_plot = [xi[0] for xi in path.x]
    y_plot = [xi[1] for xi in path.x]

    if path_str == "true":
        ax.plot(x_plot, y_plot, "k--", linewidth=3)
        ax.plot(x_plot, y_plot, "*", color="k", markersize=5)
    elif path_str == "postmean":
        ax.plot(x_plot, y_plot, "r--", linewidth=3)
        ax.plot(x_plot, y_plot, "*", color="r", markersize=5)
    elif path_str == "samp":
        ax.plot(x_plot, y_plot, "k--", linewidth=1, alpha=0.3)
        ax.plot(x_plot, y_plot, "o", alpha=0.3)
    goal = WEIRD_GAIN_GOAL
    ax.scatter(goal[0], goal[1], color="green", s=100, zorder=99)
    return ax, fig


def scatter(ax, x, **kwargs):
    x = np.atleast_2d(np.array(x))
    if x.shape[1] % 2 == 1:
        x = np.concatenate([x, np.zeros((x.shape[0], 1))], axis=1)
    try:
        axes = list(ax)
        for i, ax in enumerate(axes):
            ax.scatter(x[:, i * 2], x[:, i * 2 + 1], **kwargs)
    except TypeError:
        ax.scatter(x[:, 0], x[:, 1], **kwargs)


def plot(ax, x, shape, **kwargs):
    x = np.atleast_2d(np.array(x))
    if x.shape[1] % 2 == 1:
        x = np.concatenate([x, np.zeros([x.shape[0], 1])], axis=1)
    try:
        axes = list(ax)
        for i, ax in enumerate(axes):
            ax.plot(x[:, i * 2], x[:, i * 2 + 1], shape, **kwargs)
    except TypeError:
        ax.plot(x[:, 0], x[:, 1], shape, **kwargs)


def plot_pilco_cartpole(
    path, ax=None, fig=None, domain=None, path_str="samp", env=None
):
    """Plot a path through an assumed two-dimensional state space."""
    assert path_str in ["samp", "true", "postmean"]
    if ax is None:
        assert domain is not None
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        ax.set(
            xlim=(-3, 3),
            ylim=(-0.7, 0.7),
            xlabel="$x$",
            ylabel="$y$",
        )
        if path is None:
            return ax, fig

    xall = np.array(path.x)[:, :-1]
    try:
        xall = env.unnormalize_obs(xall)
    except:
        pass
    pole_pos = get_pole_pos(xall)
    x_plot = pole_pos[:, 0]
    y_plot = pole_pos[:, 1]

    if path_str == "true":
        ax.plot(x_plot, y_plot, "k--", linewidth=3)
        ax.plot(x_plot, y_plot, "*", color="k", markersize=5)
    elif path_str == "postmean":
        ax.plot(x_plot, y_plot, "r--", linewidth=3)
        ax.plot(x_plot, y_plot, "*", color="r", markersize=5)
    # elif path_str == "samp":
    # ax.plot(x_plot, y_plot, 'k--', linewidth=1, alpha=0.3)
    # ax.plot(x_plot, y_plot, 'o', alpha=0.3)
    elif path_str == "samp":
        lines2d = ax.plot(x_plot, y_plot, "--", linewidth=1, alpha=0.3)
        ax.plot(x_plot, y_plot, "o", color=lines2d[0].get_color(), alpha=0.3)

    # Also plot small indicator of start-of-path
    ax.plot(x_plot[0], y_plot[0], "<", markersize=2, color="k", alpha=0.5)

    return ax, fig


def plot_cartpole(path, ax=None, domain=None, path_str="samp"):
    """Plot a path through an assumed two-dimensional state space."""
    assert path_str in ["samp", "true", "postmean"]
    if ax is None:
        assert domain is not None
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        ax.set(
            xlim=(domain[0][0], domain[0][1]),
            ylim=(domain[2][0], domain[2][1]),
            xlabel="$x$",
            ylabel="$\\theta$",
        )

    x_plot = [xi[0] for xi in path.x]
    y_plot = [xi[2] for xi in path.x]

    if path_str == "true":
        ax.plot(x_plot, y_plot, "k--", linewidth=3)
        ax.plot(x_plot, y_plot, "*", color="k", markersize=5)
    elif path_str == "postmean":
        ax.plot(x_plot, y_plot, "r--", linewidth=3)
        ax.plot(x_plot, y_plot, "*", color="r", markersize=5)
    elif path_str == "samp":
        ax.plot(x_plot, y_plot, "k--", linewidth=1, alpha=0.3)
        ax.plot(x_plot, y_plot, "o", alpha=0.3)
    return ax


def plot_acrobot(path, ax=None, domain=None, path_str="samp", env=None):
    """Plot a path through an assumed two-dimensional state space."""
    assert path_str in ["samp", "true", "postmean"]
    if ax is None:
        assert domain is not None
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        ax.set(
            xlim=(domain[0][0], domain[0][1]),
            ylim=(domain[1][0], domain[1][1]),
            xlabel="$\\theta_1$",
            ylabel="$\\theta_2$",
        )

    x_plot = [xi[0] for xi in path.x]
    y_plot = [xi[1] for xi in path.x]

    if path_str == "true":
        ax.plot(x_plot, y_plot, "k--", linewidth=3)
        ax.plot(x_plot, y_plot, "*", color="k", markersize=5)
    elif path_str == "postmean":
        ax.plot(x_plot, y_plot, "r--", linewidth=3)
        ax.plot(x_plot, y_plot, "*", color="r", markersize=5)
    elif path_str == "samp":
        ax.plot(x_plot, y_plot, "k--", linewidth=1, alpha=0.3)
        ax.plot(x_plot, y_plot, "o", alpha=0.3)
    return ax


def noop(*args, ax=None, fig=None, **kwargs):
    return (
        ax,
        fig,
    )


def make_plot_obs(data, env, normalize_obs):
    obs_dim = env.observation_space.low.size
    x_data = np.array(data)
    if normalize_obs:
        norm_obs = x_data[..., :obs_dim]
        action = x_data[..., obs_dim:]
        unnorm_obs = env.unnormalize_obs(norm_obs)
        action = x_data[..., obs_dim:]
        unnorm_action = env.unnormalize_action(action)
        x_data = np.concatenate([unnorm_obs, unnorm_action], axis=-1)
    return x_data
