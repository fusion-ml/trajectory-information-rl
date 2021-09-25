python run.py -m name=bac_beta_tracking alg=bac num_iters=300 eval_frequency=5 env=beta_tracking seed="range(5)" hydra/launcher=joblib
python run.py -m name=mbrl_beta_tracking alg=mbrl num_iters=300 eval_frequency=5 env=beta_tracking seed="range(5)" hydra/launcher=joblib
python run.py -m name=random_beta_tracking alg=random num_iters=800 eval_frequency=50 env=beta_tracking seed="range(5)" hydra/launcher=joblib
