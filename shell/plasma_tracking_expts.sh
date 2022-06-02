# python run.py -m name=bac_plasma_tracking alg=bac num_iters=1200 eval_frequency=5 env=plasma_tracking seed="range(5)" hydra/launcher=joblib
# python run.py -m name=mbrl_plasma_tracking alg=mbrl num_iters=1200 eval_frequency=50 env=plasma_tracking seed="range(5)" hydra/launcher=joblib
# python run.py -m name=random_plasma_tracking alg=random num_iters=1200 eval_frequency=50 env=plasma_tracking seed="range(5)" hydra/launcher=joblib
# python run.py -m name=us_plasma_tracking alg=us num_iters=1200 eval_frequency=50 env=plasma_tracking seed="range(5)" hydra/launcher=joblib
python run.py -m name=tip_plasma_tracking alg=rollout_barl num_iters=1000 eval_frequency=50 env=plasma_tracking seed="range(5)" hydra/launcher=joblib
