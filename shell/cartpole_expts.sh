python run.py -m name=bac_cartpole alg=bac num_iters=300 eval_frequency=10 env=cartpole seed="range(5)" hydra/launcher=joblib
python run.py -m name=mbrl_cartpole alg=mbrl num_iters=300 eval_frequency=10 env=cartpole seed="range(5)" hydra/launcher=joblib
python run.py -m name=random_cartpole alg=random num_iters=300 eval_frequency=10 env=cartpole seed="range(5)" hydra/launcher=joblib
