# bayesian-active-control
Bayesian active control with BAX


## Installation

To install dependencies for BAX, `cd` into this repo directory and run:
```bash
$ pip install -r requirements/requirements_bax.txt
$ pip install -r requirements/requirements_bax_gpfs.txt
```

For some functionality, you'll need to compile a [Stan](https://mc-stan.org/) model by
running:
```bash
$ python bax/models/stan/compile_models.py
```

## Running Examples

Willie is in the process of adding more examples. For now, see
`examples/es/00_bax_viz2d_simple_demo.py`.

First make sure this repo directory is on the PYTHONPATH, e.g. by running:
```bash
$ source shell/add_pwd_to_pythonpath.sh
```

And then run:
```bash
$ python examples/es/00_bax_viz2d_simple_demo.py
```
