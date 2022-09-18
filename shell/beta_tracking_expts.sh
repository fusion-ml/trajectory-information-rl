python run.py -m name=tip_beta_tracking alg=tip num_iters=800 eval_frequency=5 env=beta_tracking seed="range(5)" hydra/launcher=joblib
python run.py -m name=barl_beta_tracking alg=barl num_iters=800 eval_frequency=5 env=beta_tracking seed="range(5)" hydra/launcher=joblib
python run.py -m name=mpc_beta_tracking alg=mpc num_iters=800 eval_frequency=5 env=beta_tracking seed="range(5)" hydra/launcher=joblib
python run.py -m name=eigt_beta_tracking alg=eigt num_iters=800 eval_frequency=50 env=beta_tracking seed="range(5)" hydra/launcher=joblib
python run.py -m name=stip_beta_tracking alg=stip num_iters=800 eval_frequency=5 env=beta_tracking seed="range(5)" hydra/launcher=joblib
python run.py -m name=sdip_barl_beta_tracking alg=sdip num_iters=800 eval_frequency=5 env=beta_tracking seed="range(5)" hydra/launcher=joblib
python run.py -m name=dip_barl_beta_tracking alg=dip num_iters=800 eval_frequency=5 env=beta_tracking seed="range(5)" hydra/launcher=joblib
