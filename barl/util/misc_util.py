"""
Miscellaneous utilities.
"""

from argparse import Namespace
from pathlib import Path
import os
import numpy as np
import logging
import pickle
from collections import defaultdict


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
        logging.info(f'Dumper dumping to {cwd}')
        self.info = defaultdict(list)
        self.info_path = self.expdir / 'info.pkl'

    def add(self, name, val, verbose=True, log_mean_std=False):
        if verbose:
            try:
                val = float(val)
                logging.info(f"{name}: {val:.3f}")
            except TypeError:
                logging.info(f"{name}: {val}")
            if log_mean_std:
                valarray = np.array(val)
                logging.info(f"{name}: mean={valarray.mean():.3f} std={valarray.std():.3f}")
        self.info[name].append(val)

    def extend(self, name, vals, verbose=False):
        if verbose:
            logging.info(f"{name}: {vals}")
        self.info[name].extend(vals)

    def save(self):
        with self.info_path.open('wb') as f:
            pickle.dump(self.info, f)


def batch_function(f):
    # naively batch a function by calling it on each element separately and making a list of those
    def batched_f(x_list):
        y_list = []
        for x in x_list:
            y_list.append(f(x))
        return y_list
    return batched_f


def make_postmean_fn(model):
    def postmean_fn(x):
        mu_list, std_list = model.get_post_mu_cov(x, full_cov=False)
        mu_tup_for_x = list(zip(*mu_list))
        return mu_tup_for_x
    return postmean_fn
