# python run.py -m name=rollout_reacher alg=rollout_barl num_iters=1000 eval_frequency=25 env=reacher seed="range(5)" hydra/launcher=joblib
# python run.py -m name=bac_reacher alg=barl num_iters=1500 eval_frequency=50 env=reacher seed="range(5)" hydra/launcher=joblib
# python run.py -m name=mbrl_reacher alg=mbrl num_iters=1500 eval_frequency=50 env=reacher seed="range(5)" hydra/launcher=joblib
# python run.py -m name=us_reacher alg=us num_iters=1500 eval_frequency=50 env=reacher seed="range(5)" hydra/launcher=joblib
python run.py -m name=sum_barl_reacher alg=sum_barl num_iters=1000 eval_frequency=50 env=reacher seed="range(5)" hydra/launcher=joblib
# python run.py -m name=sus_reacher alg=sus num_iters=1500 eval_frequency=50 env=reacher seed="range(5)" hydra/launcher=joblib
# python run.py -m name=rus_reacher alg=rus num_iters=1500 eval_frequency=50 env=reacher seed="range(5)" hydra/launcher=joblib
