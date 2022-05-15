python run.py -m name=open_loop_bac_beta_tracking_fixed alg=open_loop_barl num_iters=200 eval_frequency=15 env=beta_tracking_fixed  seed="range(5)" hydra/launcher=joblib
python run.py -m name=open_loop_mpc_beta_tracking_fixed alg=open_loop_mpc num_iters=200 eval_frequency=15 env=beta_tracking_fixed  seed="range(5)" hydra/launcher=joblib
python run.py -m name=open_loop_us_beta_tracking_fixed alg=open_loop_us num_iters=200 eval_frequency=15 env=beta_tracking_fixed  seed="range(5)" hydra/launcher=joblib
