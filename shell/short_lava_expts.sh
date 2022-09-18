python run.py -m name=otip_short_lava alg=otip num_iters=200 eval_frequency=20 env=short_lava_path eigmpc.num_iters=6 test_mpc.num_iters=6 seed="range(5)" hydra/launcher=joblib
python run.py -m name=open_loop_mpc_short_lava alg=open_loop_mpc num_iters=200 eval_frequency=20 env=short_lava_path eigmpc.num_iters=6 test_mpc.num_iters=6 seed="range(5)" hydra/launcher=joblib
python run.py -m name=open_loop_us_short_lava alg=open_loop_us num_iters=200 eval_frequency=20 env=short_lava_path eigmpc.num_iters=6 test_mpc.num_iters=6 seed="range(5)" hydra/launcher=joblib
