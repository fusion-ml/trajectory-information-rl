python run.py -m name=open_loop_bac_weird_gain alg=open_loop_barl num_iters=200 eval_frequency=10 env=weird_gain eigmpc.num_iters=6 test_mpc.num_iters=6 seed="range(5)" hydra/launcher=joblib
python run.py -m name=open_loop_mpc_weird_gain alg=open_loop_mpc num_iters=200 eval_frequency=10 env=weird_gain eigmpc.num_iters=6 test_mpc.num_iters=6 seed="range(5)" hydra/launcher=joblib
python run.py -m name=open_loop_us_weird_gain alg=open_loop_us num_iters=200 eval_frequency=10 env=weird_gain eigmpc.num_iters=6 test_mpc.num_iters=6 seed="range(5)" hydra/launcher=joblib
