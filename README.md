# bayesian-active-control
Bayesian active control with BAX

## Installation

To install dependencies for BAX, `cd` into this repo and run:
```bash
$ pip install -r requirements/requirements_bax.txt
$ pip install -r requirements/requirements_bax_gpfs.txt
```

For some functionality, you'll need to compile a [Stan](https://mc-stan.org/) model by
running:
```bash
$ python bax/models/stan/compile_models.py
```
