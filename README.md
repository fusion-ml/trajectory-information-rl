# Bayesian Active Reinforcement Learning

## Installation

To install dependencies for this code, make a virtual environment with a modern
Python version and run
```bash
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
The fusion dependencies are a bit more involved and rely on proprietary data so we have not included them here. A full release of the code is forthcoming. The 
