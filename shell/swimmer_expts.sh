python run.py -m name=bac_swimmer alg=bac num_iters=1500 eval_frequency=50 env=swimmer seed="range(5)" hydra/launcher=joblib
python run.py -m name=mbrl_swimmer alg=mbrl num_iters=1500 eval_frequency=50 env=swimmer seed="range(5)" hydra/launcher=joblib
python run.py -m name=random_swimmer alg=random num_iters=1500 eval_frequency=50 env=swimmer seed="range(5)" hydra/launcher=joblib
python run.py -m name=us_swimmer alg=us num_iters=1500 eval_frequency=50 env=swimmer seed="range(5)" hydra/launcher=joblib
