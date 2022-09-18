python run.py -m name=tip_reacher alg=tip num_iters=1000 eval_frequency=25 env=reacher seed="range(5)" hydra/launcher=joblib
python run.py -m name=barl_reacher alg=barl num_iters=1500 eval_frequency=50 env=reacher seed="range(5)" hydra/launcher=joblib
python run.py -m name=mpc_reacher alg=mpc num_iters=1500 eval_frequency=50 env=reacher seed="range(5)" hydra/launcher=joblib
python run.py -m name=eigt_reacher alg=eigt num_iters=1500 eval_frequency=50 env=reacher seed="range(5)" hydra/launcher=joblib
python run.py -m name=stip_reacher alg=stip num_iters=1500 eval_frequency=50 env=reacher seed="range(5)" hydra/launcher=joblib
python run.py -m name=sdip_reacher alg=sdip num_iters=1500 eval_frequency=50 env=reacher seed="range(5)" hydra/launcher=joblib
python run.py -m name=dip_reacher alg=dip num_iters=1500 eval_frequency=50 env=reacher seed="range(5)" hydra/launcher=joblib
