# Bayesian Active Reinforcement Learning

## Installation

To install dependencies for this code, make a virtual environment with a modern
Python version (>= 3.8) and install the requirements as follows:
```bash
$ python -m venv venvbarl
$ source venvbarl/bin/activate
$ pip install -r requirements.txt
```

## Running Experiments
We have script files inside the `shell/` directory that reproduce the experiments built off of this codebase.
Those are the methods TIP, sTIP, BARL, MPC, EIG_T, DIP, and sDIP from the paper. We will release a separate repository that uses the environments in this one
inside `bax/envs/` but runs PETS, SAC, and PPO.
To reproduce these, run
```bash
$ ./shell/{exp_name}_expts.sh
```
The `exp_name` should be one of `reacher`, `pendulum`, or `cartpole`.
The fusion dependencies are a bit more involved and rely on proprietary data so we have not included them here.

## Adding a New Environment
One obvious use of this codebase is to compare to subsequent algorithms in order to ascertain their relative performance.
A researcher will likely want to add environments for running in this setup. BARL and TIP have somewhat finicky requirements for this so
we detail them here:
- We build off of the OpenAI gym interface with a couple modifications.
- We assume the gym environment has an attribute `self.horizon` that is the integer value for how long episodes should take.
- We assume (for BARL in the TQRL setting and for gathering dyamics data with which to evaluate model accuracy) that `env.reset(obs)` resets the env to a start at state `obs` for a numpy array `obs` in the observation space of the environment. If you need to add an environment that can't support this interface, we would recommend simply commenting out the code that uses it.
Finally, you need to register the environment and add it to the config. There are many examples of registration in `barl/envs/__init__.py` and these are quite simple to follow. There are similarly many examples in `cfg/env/` of config files for new environments. However, these include GP kernel hyperparameters and MPC hyperparameters for the controller for your environment. Next, we'll discuss fitting the GP kernel parameters and finding a good controller setting.

### Tuning GP Kernel Parameters
We have included a script for tuning GP kernel parameters as part of `run.py` via the flag `fit_gp_hypers=True`. Typically, 1500 or so randomly selected points from the domain work well to find a kernel. It is easy to modify the code to accept data from disk instead of generating it online. For an example invocation, see the file `shell/hyper_fit_cartpole.sh`.

### Finding an appropriate controller setting
As we are using zeroth order stochastic optimization algorithms for our controller and in particular the [iCEM algorithm](https://is.mpg.de/publications/pinnerietal2020-icem), we have several parameters which increase the computational cost of the controller but may result in a better action sequence. These are the base number of samples `nsamps` per iteration, the planning horizon `planning_horizon`, the number of elites `n_elites`, the number of iterations `num_iters`, and the number of actions to execute before replanning `actions_per_plan`. In general larger numbers of samples, a number of elites roughly 10%-20% of the number of samples, a larger number of iterations, a medium planning horizon (10 or so), and a small number of actions between each replanning step lead to the best performance if the time can be spared.

Example configurations that match those used in experiments in our paper can be found in the environment config files in `cfg/env/*.yaml`. We recommend starting with similar values and changing them to be more accurate or faster as needed. The code by default runs a handful of episodes on the ground truth dynamics using the same control settings, so if that doesn't reach the desired level of performance we recommend increasing the controller budget.
