python run.py -m name=rollout_bac_weird_gain alg=rollout_barl num_iters=50 eval_frequency=2 env=weird_gain seed="range(5)" hydra/launcher=joblib num_samples_mc=3
python run.py -m name=ss_bac_weird_gain alg=start_state_barl num_iters=50 eval_frequency=2 env=weird_gain seed="range(5)" hydra/launcher=joblib num_samples_mc=3
python run.py -m name=open_loop_bac_weird_gain alg=open_loop_barl num_iters=50 eval_frequency=10 env=weird_gain seed="range(5)" hydra/launcher=joblib
python run.py -m name=barl_weird_gain alg=barl num_iters=50 eval_frequency=2 env=weird_gain seed="range(5)" hydra/launcher=joblib
python run.py -m name=mpc_weird_gain alg=mpc num_iters=50 eval_frequency=2 env=weird_gain # seed="range(5)" hydra/launcher=joblib
# python run.py -m name=random_weird_gain alg=random num_iters=1500 eval_frequency=50 env=weird_gain seed="range(5)" hydra/launcher=joblib
python run.py -m name=us_weird_gain alg=us num_iters=50 eval_frequency=2 env=weird_gain # seed="range(5)" hydra/launcher=joblib
