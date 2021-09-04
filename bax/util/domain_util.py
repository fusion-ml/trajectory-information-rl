"""
Utilities for domains (search spaces).
"""

import numpy as np


def unif_random_sample_domain(n=1, domain=None):
    """Draws a sample uniformly at random from domain (a list of tuple bounds)."""
    list_of_arr_per_dim = [np.random.uniform(dom[0], dom[1], n) for dom in domain]
    list_of_list_per_sample = [list(l) for l in np.array(list_of_arr_per_dim).T]
    return list_of_list_per_sample


def unif_random_sample_cylinder(n=1, domain=None, env=None):
    periodic_dimensions = env.periodic_dimensions
    data = []
    for i, dom in enumerate(domain):
        if i in periodic_dimensions:
            theta = np.random.uniform(-np.pi, np.pi, n)
            data.append(np.sin(theta))
            data.append(np.cos(theta))
            continue
        if i - 1 in periodic_dimensions:
            continue
        data.append(np.random.uniform(dom[0], dom[1], n))
    list_of_arr_per_dim = data
    list_of_list_per_sample = [list(l) for l in np.array(list_of_arr_per_dim).T]
    return list_of_list_per_sample


def project_to_domain(x, domain):
    """Project x, a list of scalars, to be within domain (a list of tuple bounds)."""

    # Assume input x is either a list or 1d numpy array
    assert isinstance(x, list) or isinstance(x, np.ndarray)
    if isinstance(x, np.ndarray):
        assert len(x.shape) == 1
        x_is_list = False
    else:
        x_is_list = True

    # Project x to be within domain
    x_arr = np.array(x)
    min_list = [tup[0] for tup in domain]
    max_list = [tup[1] for tup in domain]
    x_arr_clip = np.clip(x_arr, min_list, max_list)

    # Convert to original type (either list or keep as 1d numpy array)
    x_return = list(x_arr_clip) if x_is_list else x_arr_clip

    return x_return


def project_to_cylinder(x, domain, env):
    x = np.array(x)
    for i, dim in enumerate(env.periodic_dimensions):
        true_dim = i + dim
        norm = np.sqrt(np.sum(np.square(x[..., i:i+2]), axis=-1))
        x[..., i:i+2] /= norm
    projected_x = project_to_domain(x)
    return projected_x
