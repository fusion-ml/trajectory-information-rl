python run.py -m name=open_loop_bac_weirder_gain alg=open_loop_barl num_iters=200 eval_frequency=10 env=weirder_gain  seed="range(5)" hydra/launcher=joblib
python run.py -m name=open_loop_mpc_weirder_gain alg=open_loop_mpc num_iters=200 eval_frequency=10 env=weirder_gain  seed="range(5)" hydra/launcher=joblib
python run.py -m name=open_loop_us_weirder_gain alg=open_loop_us num_iters=200 eval_frequency=10 env=weirder_gain  seed="range(5)" hydra/launcher=joblib
