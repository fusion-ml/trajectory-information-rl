import numpy as np
from tqdm import trange


def CEM(start_state, obs_dim, action_dim, dynamics_fn, horizon, alpha, popsize, elite_frac, num_iters, verbose=False):
    '''
    CEM: the cross-entropy method, here used for planning optimal actions on an MDP.
    assumes action space is [-1, 1]^action_dim
    '''
    action_upper_bound = 1
    action_lower_bound = -1
    initial_variance_divisor = 4
    num_elites = int(popsize * elite_frac)
    mean = np.zeros(action_dim)
    var = np.ones_like(mean) * ((action_upper_bound - action_lower_bound) / initial_variance_divisor) ** 2
    best_sample, best_return = None, -np.inf
    for i in trange(num_iters, disable=not verbose):
        samples = np.fmod(np.random.normal(size=(popsize, horizon, action_dim)), 2) * np.sqrt(var) + mean
        samples = np.clip(samples, action_lower_bound, action_upper_bound)
        returns = dynamics_fn(start_state, samples)
        elites = samples[np.argsort(returns)[-num_elites:], ...]
        new_mean = np.mean(elites, axis=0)
        new_var = np.var(elites, axis=0)
        mean = alpha * mean + (1 - alpha) * new_mean
        var = alpha * var + (1 - alpha) * new_var
        best_idx = np.argmax(returns)
        best_current_return = returns[best_idx]
        if best_current_return > best_return:
            best_return = best_current_return
            best_sample = samples[best_idx, ...]
    return best_return, best_sample
