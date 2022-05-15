python run.py -m name=rollout_cartpole alg=rollout_barl num_iters=300 eval_frequency=10 env=cartpole seed="range(5)" hydra/launcher=joblib
# python run.py -m name=bac_cartpole alg=bac num_iters=300 eval_frequency=10 env=cartpole seed="range(5)" hydra/launcher=joblib
# python run.py -m name=mbrl_cartpole alg=mbrl num_iters=300 eval_frequency=10 env=cartpole seed="range(5)" hydra/launcher=joblib
# python run.py -m name=random_cartpole alg=random num_iters=300 eval_frequency=10 env=cartpole seed="range(5)" hydra/launcher=joblib
# python run.py -m name=us_cartpole alg=us num_iters=300 eval_frequency=10 env=cartpole seed="range(5)" hydra/launcher=joblib
# python run.py -m name=sum_barl_cartpole alg=sum_barl num_iters=300 eval_frequency=10 env=cartpole seed="range(5)" hydra/launcher=joblib
python run.py -m name=sus_cartpole alg=sus num_iters=300 eval_frequency=10 env=cartpole seed="range(5)" hydra/launcher=joblib
python run.py -m name=rus_cartpole alg=rus num_iters=300 eval_frequency=10 env=cartpole seed="range(5)" hydra/launcher=joblib
