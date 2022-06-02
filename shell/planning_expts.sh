python run.py -m name=tip_beta_rotation_tracking_ph2 alg=rollout_barl num_iters=500 eval_frequency=20 env=new_beta_rotation seed="range(5)" hydra/launcher=joblib env.eigmpc.planning_horizon=2 &
python run.py -m name=tip_beta_rotation_tracking_ph4 alg=rollout_barl num_iters=500 eval_frequency=20 env=new_beta_rotation seed="range(5)" hydra/launcher=joblib env.eigmpc.planning_horizon=4 &
python run.py -m name=tip_beta_rotation_tracking_ph8 alg=rollout_barl num_iters=500 eval_frequency=20 env=new_beta_rotation seed="range(5)" hydra/launcher=joblib env.eigmpc.planning_horizon=8 &
python run.py -m name=tip_beta_rotation_tracking_ph10 alg=rollout_barl num_iters=500 eval_frequency=20 env=new_beta_rotation seed="range(5)" hydra/launcher=joblib env.eigmpc.planning_horizon=10 &
python run.py -m name=tip_beta_rotation_tracking_ph16 alg=rollout_barl num_iters=500 eval_frequency=20 env=new_beta_rotation seed="range(5)" hydra/launcher=joblib env.eigmpc.planning_horizon=16 &
