python run.py -m name=rollout_reacher alg=rollout_bac num_iters=1000 eval_frequency=25 env=reacher seed="range(5)" hydra/launcher=joblib
# python run.py -m name=bac_reacher alg=bac num_iters=1500 eval_frequency=50 env=reacher seed="range(5)" hydra/launcher=joblib
# python run.py -m name=mbrl_reacher alg=mbrl num_iters=1500 eval_frequency=50 env=reacher seed="range(5)" hydra/launcher=joblib
# python run.py -m name=random_reacher alg=random num_iters=1500 eval_frequency=50 env=reacher seed="range(5)" hydra/launcher=joblib
# python run.py -m name=us_reacher alg=us num_iters=1500 eval_frequency=50 env=reacher seed="range(5)" hydra/launcher=joblib
