python run.py -m name=bac_plasma_tracking alg=bac num_iters=300 eval_frequency=5 env=plasma_tracking seed="range(5)" hydra/launcher=joblib
python run.py -m name=mbrl_plasma_tracking alg=mbrl num_iters=300 eval_frequency=5 env=plasma_tracking seed="range(5)" hydra/launcher=joblib
python run.py -m name=random_plasma_tracking alg=random num_iters=800 eval_frequency=50 env=plasma_tracking seed="range(5)" hydra/launcher=joblib
