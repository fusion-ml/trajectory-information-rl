python run.py -m name=rollout_pendulum alg=rollout_barl num_iters=200 eval_frequency=5 env=pendulum seed="range(5)" hydra/launcher=joblib
python run.py -m name=bac_pendulum alg=bac num_iters=200 eval_frequency=5 env=pendulum seed="range(5)" hydra/launcher=joblib
python run.py -m name=mbrl_pendulum alg=mbrl num_iters=200 eval_frequency=5 env=pendulum seed="range(5)" hydra/launcher=joblib
python run.py -m name=us_pendulum alg=usnum_iters=200 eval_frequency=5 env=pendulum seed="range(5)" hydra/launcher=joblib
python run.py -m name=sum_barl_pendulum alg=sum_barl num_iters=200 eval_frequency=5 env=pendulum seed="range(5)" hydra/launcher=joblib
python run.py -m name=sus_pendulum alg=sus num_iters=200 eval_frequency=5 env=pendulum seed="range(5)" hydra/launcher=joblib
python run.py -m name=rus_pendulum alg=rus num_iters=200 eval_frequency=5 env=pendulum seed="range(5)" hydra/launcher=joblib
