python run.py -m name=open_loop_bac_beta_tracking alg=open_loop_barl num_iters=200 eval_frequency=30 env=beta_tracking eigmpc.num_iters=6 test_mpc.num_iters=6 seed="range(5)" hydra/launcher=joblib
python run.py -m name=open_loop_mpc_beta_tracking alg=open_loop_mpc num_iters=200 eval_frequency=30 env=beta_tracking eigmpc.num_iters=6 test_mpc.num_iters=6 seed="range(5)" hydra/launcher=joblib
python run.py -m name=open_loop_us_beta_tracking alg=open_loop_us num_iters=200 eval_frequency=30 env=beta_tracking eigmpc.num_iters=6 test_mpc.num_iters=6 seed="range(5)" hydra/launcher=joblib
