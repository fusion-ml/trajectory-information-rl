"""
Miscellaneous utilities.
"""

from argparse import Namespace
from pathlib import Path
import os
import numpy as np
import tensorflow as tf
import logging
import pickle
from collections import defaultdict
from scipy.stats import norm


def dict_to_namespace(params):
    """
    If params is a dict, convert it to a Namespace, and return it.

    Parameters ----------
    params : Namespace_or_dict
        Namespace or dict.

    Returns
    -------
    params : Namespace
        Namespace of params
    """
    # If params is a dict, convert to Namespace
    if isinstance(params, dict):
        params = Namespace(**params)

    return params


class suppress_stdout_stderr:
    """
    A context manager for doing a "deep suppression" of stdout and stderr in
    Python, i.e. will suppress all print, even if the print originates in a
    compiled C/Fortran sub-function.

    Source: https://stackoverflow.com/q/11130156
    """

    def __init__(self):
        # Open a pair of null files
        self.null_fds = [os.open(os.devnull, os.O_RDWR) for x in range(2)]

        # Save the actual stdout (1) and stderr (2) file descriptors.
        self.save_fds = [os.dup(1), os.dup(2)]

    def __enter__(self):
        # Assign the null pointers to stdout and stderr.
        os.dup2(self.null_fds[0], 1)
        os.dup2(self.null_fds[1], 2)

    def __exit__(self, *_):
        # Re-assign the real stdout/stderr back to (1) and (2)
        os.dup2(self.save_fds[0], 1)
        os.dup2(self.save_fds[1], 2)

        # Close the null files
        for fd in self.null_fds + self.save_fds:
            os.close(fd)


class Dumper:
    def __init__(self, experiment_name):
        cwd = Path.cwd()
        # this should be the root of the repo
        self.expdir = cwd
        logging.info(f"Dumper dumping to {cwd}")
        self.info = defaultdict(list)
        self.info_path = self.expdir / "info.pkl"

    def add(self, name, val, verbose=True, log_mean_std=False):
        if verbose:
            try:
                val = float(val)
                logging.info(f"{name}: {val:.3f}")
            except TypeError:
                logging.info(f"{name}: {val}")
            if log_mean_std:
                valarray = np.array(val)
                logging.info(
                    f"{name}: mean={valarray.mean():.3f} std={valarray.std():.3f}"
                )
        self.info[name].append(val)

    def extend(self, name, vals, verbose=False):
        if verbose:
            disp_vals = [f"{val:.3f}" for val in vals]
            logging.info(f"{name}: {disp_vals}")
        self.info[name].extend(vals)

    def save(self):
        with self.info_path.open("wb") as f:
            pickle.dump(self.info, f)


def batch_function(f):
    # naively batch a function by calling it on each element separately and making a list of those
    def batched_f(x_list):
        y_list = []
        for x in x_list:
            y_list.append(f(x))
        return y_list

    return batched_f


def make_postmean_fn(model, use_tf=False):
    def postmean_fn(x):
        mu_list, std_list = model.get_post_mu_cov(x, full_cov=False)
        mu_list = np.array(mu_list)
        mu_tup_for_x = list(zip(*mu_list))
        return mu_tup_for_x

    if not use_tf:
        return postmean_fn

    def tf_postmean_fn(x):
        x = tf.convert_to_tensor(x, dtype=tf.float32)
        mu_list, std_list = model.get_post_mu_cov(x, full_cov=False)
        mu_tup_for_x = list(mu_list.numpy())
        return mu_tup_for_x

    return tf_postmean_fn


def mse(y, y_hat):
    y = np.array(y)
    y_hat = np.array(y_hat)
    return np.mean(np.sum(np.square(y_hat - y), axis=1))


def model_likelihood(model, x, y):
    """
    assume x is list of n d_x-dim ndarrays
    and y is list of n d_y-dim ndarrays
    """
    # mu should be list of d_y n-dim ndarrays
    # cov should be list of d_y n-dim ndarrays
    n = len(x)
    mu, cov = model.get_post_mu_cov(x)
    y = np.array(y).flatten()
    mu = np.array(mu).T.flatten()
    cov = np.array(cov).T.flatten()
    white_y = (y - mu) / np.sqrt(cov)
    logpdfs = norm.logpdf(white_y)
    logpdfs = logpdfs.reshape((n, -1))
    avg_likelihood = logpdfs.sum(axis=1).mean()
    return avg_likelihood


def get_tf_dtype(precision):
    if precision == 32:
        return tf.float32
    elif precision == 64:
        return tf.float64
    else:
        raise ValueError(f"TF Precision {precision} not supported")


def flatten(policy_list):
    out = []
    for item in policy_list:
        if type(item) is list:
            out += item
        else:
            out.append(item)
    return out
