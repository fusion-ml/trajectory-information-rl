"""
Miscellaneous utilities.
"""

from argparse import Namespace
from pathlib import Path
import os
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
        # while cwd.name != 'bayesian-active-control':
            # cwd = cwd.parent
        # this should be the root of the repo
        self.expdir = cwd
        logging.info(f'Dumper dumping to {cwd}')
        # if self.expdir.exists() and overwrite:
            # shutil.rmtree(self.expdir)
        # self.expdir.mkdir(parents=True)
        self.info = defaultdict(list)
        self.info_path = self.expdir / 'info.pkl'
        # args = vars(args)
        # print('Run with the following args:')
        # pprint(args)
        # args_path = self.expdir / 'args.json'
        # with args_path.open('w') as f:
            # json.dump(args, f, indent=4)

    def add(self, name, val):
        self.info[name].append(val)

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
