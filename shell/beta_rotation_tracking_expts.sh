# python run.py -m name=tip_beta_rotation_tracking alg=rollout_barl num_iters=500 eval_frequency=20 env=new_beta_rotation seed="range(5)" hydra/launcher=joblib
# python run.py -m name=barl_beta_rotation_tracking alg=barl num_iters=500 eval_frequency=20 env=new_beta_rotation seed="range(5)" hydra/launcher=joblib
# python run.py -m name=mpc_beta_rotation_tracking alg=mpc num_iters=500 eval_frequency=20 env=new_beta_rotation seed="range(5)" hydra/launcher=joblib
# python run.py -m name=random_beta_rotation_tracking alg=random num_iters=500 eval_frequency=20 env=new_beta_rotation seed="range(5)" hydra/launcher=joblib
# python run.py -m name=us_beta_rotation_tracking alg=us num_iters=1000 eval_frequency=20 env=new_beta_rotation seed="range(5)" hydra/launcher=joblib
# python run.py -m name=sum_barl_beta_rotation_tracking alg=sum_barl num_iters=500 eval_frequency=20 env=new_beta_rotation seed="range(5)" hydra/launcher=joblib
python run.py alg=sum_barl num_iters=500 eval_frequency=20 env=new_beta_rotation
