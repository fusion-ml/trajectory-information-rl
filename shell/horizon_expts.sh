# python run.py -m name=rollout_beta_tracking_ph2 alg=rollout_barl num_iters=1000 eval_frequency=50 env=beta_tracking seed="range(5)" hydra/launcher=joblib env.eigmpc.planning_horizon=2
python run.py -m name=rollout_beta_tracking_ph4 alg=rollout_barl num_iters=500 eval_frequency=10 env=beta_tracking seed="range(5)" hydra/launcher=joblib env.eigmpc.planning_horizon=4
python run.py -m name=rollout_beta_tracking_ph6 alg=rollout_barl num_iters=500 eval_frequency=10 env=beta_tracking seed="range(5)" hydra/launcher=joblib env.eigmpc.planning_horizon=6
python run.py -m name=rollout_beta_tracking_ph8 alg=rollout_barl num_iters=500 eval_frequency=10 env=beta_tracking seed="range(5)" hydra/launcher=joblib env.eigmpc.planning_horizon=8
python run.py -m name=rollout_beta_tracking_ph10 alg=rollout_barl num_iters=500 eval_frequency=10 env=beta_tracking seed="range(5)" hydra/launcher=joblib env.eigmpc.planning_horizon=10
