# python run.py -m name=us_pendulum alg=us num_iters=200 eval_frequency=5 env=pendulum seed="range(5)" hydra/launcher=joblib
# python run.py -m name=us_cartpole alg=us num_iters=300 eval_frequency=10 env=cartpole seed="range(5)" hydra/launcher=joblib
# python run.py -m name=us_lava_path alg=us num_iters=400 eval_frequency=10 env=lava_path seed="range(5)" hydra/launcher=joblib
python run.py -m name=us_reacher alg=us num_iters=1500 eval_frequency=50 env=reacher seed="range(5)" hydra/launcher=joblib
python run.py -m name=us_beta_tracking alg=us num_iters=1200 eval_frequency=50 env=beta_tracking seed="range(5)" hydra/launcher=joblib
python run.py -m name=us_plasma_tracking alg=us num_iters=1200 eval_frequency=50 env=plasma_tracking seed="range(5)" hydra/launcher=joblib
