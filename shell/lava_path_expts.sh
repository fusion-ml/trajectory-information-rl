python run.py -m name=bac_lava_path alg=bac num_iters=100 eval_frequency=5 env=lava_path seed="range(5)" hydra/launcher=joblib
python run.py -m name=mbrl_lava_path alg=mbrl num_iters=400 eval_frequency=10 env=lava_path seed="range(5)" hydra/launcher=joblib
python run.py -m name=random_lava_path alg=random num_iters=400 eval_frequency=10 env=lava_path seed="range(5)" hydra/launcher=joblib
python run.py -m name=us_lava_path alg=us num_iters=400 eval_frequency=10 env=lava_path seed="range(5)" hydra/launcher=joblib
