# python run.py -m name=rollout_beta_tracking alg=rollout_bac num_iters=800 eval_frequency=5 env=beta_tracking seed="range(5)" hydra/launcher=joblib
# python run.py -m name=bac_beta_tracking alg=bac num_iters=800 eval_frequency=5 env=beta_tracking seed="range(5)" hydra/launcher=joblib
# python run.py -m name=mbrl_beta_tracking alg=mbrl num_iters=800 eval_frequency=5 env=beta_tracking seed="range(5)" hydra/launcher=joblib
# python run.py -m name=random_beta_tracking alg=random num_iters=800 eval_frequency=50 env=beta_tracking seed="range(5)" hydra/launcher=joblib
# python run.py -m name=us_beta_tracking alg=us num_iters=800 eval_frequency=50 env=beta_tracking seed="range(5)" hydra/launcher=joblib
python run.py -m name=sum_barl_beta_tracking alg=sum_barl num_iters=800 eval_frequency=5 env=beta_tracking seed="range(5)" hydra/launcher=joblib
python run.py -m name=sus_barl_beta_tracking alg=sus num_iters=800 eval_frequency=5 env=beta_tracking seed="range(5)" hydra/launcher=joblib
python run.py -m name=rus_barl_beta_tracking alg=rus num_iters=800 eval_frequency=5 env=beta_tracking seed="range(5)" hydra/launcher=joblib
