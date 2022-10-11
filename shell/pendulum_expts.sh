python run.py -m name=tip_pendulum alg=tip num_iters=200 eval_frequency=5 env=pendulum seed="range(5)" hydra/launcher=joblib
python run.py -m name=barl_pendulum alg=barl num_iters=200 eval_frequency=5 env=pendulum seed="range(5)" hydra/launcher=joblib
python run.py -m name=mpc_pendulum alg=mpc num_iters=200 eval_frequency=5 env=pendulum seed="range(5)" hydra/launcher=joblib
python run.py -m name=us_pendulum alg=us num_iters=200 eval_frequency=5 env=pendulum seed="range(5)" hydra/launcher=joblib
python run.py -m name=stip_pendulum alg=stip num_iters=200 eval_frequency=5 env=pendulum seed="range(5)" hydra/launcher=joblib
python run.py -m name=sdip_pendulum alg=sdip num_iters=200 eval_frequency=5 env=pendulum seed="range(5)" hydra/launcher=joblib
python run.py -m name=dip_pendulum alg=dip num_iters=200 eval_frequency=5 env=pendulum seed="range(5)" hydra/launcher=joblib
