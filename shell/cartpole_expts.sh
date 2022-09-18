python run.py -m name=tip_cartpole alg=tip num_iters=300 eval_frequency=10 env=cartpole seed="range(5)" hydra/launcher=joblib
python run.py -m name=barl_cartpole alg=barl num_iters=300 eval_frequency=10 env=cartpole seed="range(5)" hydra/launcher=joblib
python run.py -m name=mpc_cartpole alg=mpc num_iters=300 eval_frequency=10 env=cartpole seed="range(5)" hydra/launcher=joblib
python run.py -m name=us_cartpole alg=us num_iters=300 eval_frequency=10 env=cartpole seed="range(5)" hydra/launcher=joblib
python run.py -m name=stip_cartpole alg=stip num_iters=300 eval_frequency=10 env=cartpole seed="range(5)" hydra/launcher=joblib
python run.py -m name=sdip_cartpole alg=sdip num_iters=300 eval_frequency=10 env=cartpole seed="range(5)" hydra/launcher=joblib
python run.py -m name=dip_cartpole alg=dip num_iters=300 eval_frequency=10 env=cartpole seed="range(5)" hydra/launcher=joblib
