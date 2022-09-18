python run.py -m name=tip_beta_rotation_tracking alg=tip num_iters=500 eval_frequency=20 env=new_beta_rotation seed="range(5)" hydra/launcher=joblib
python run.py -m name=barl_beta_rotation_tracking alg=barl num_iters=500 eval_frequency=20 env=new_beta_rotation seed="range(5)" hydra/launcher=joblib
python run.py -m name=mpc_beta_rotation_tracking alg=mpc num_iters=500 eval_frequency=20 env=new_beta_rotation seed="range(5)" hydra/launcher=joblib
python run.py -m name=eigt_beta_rotation_tracking alg=eigt num_iters=1000 eval_frequency=20 env=new_beta_rotation seed="range(5)" hydra/launcher=joblib
python run.py -m name=stip_beta_rotation_tracking alg=stip num_iters=500 eval_frequency=20 env=new_beta_rotation seed="range(5)" hydra/launcher=joblib
python run.py -m name=sdip_beta_rotation_tracking alg=sdip num_iters=500 eval_frequency=20 env=new_beta_rotation seed="range(5)" hydra/launcher=joblib
python run.py -m name=dip_beta_rotation_tracking alg=dip num_iters=500 eval_frequency=20 env=new_beta_rotation seed="range(5)" hydra/launcher=joblib
