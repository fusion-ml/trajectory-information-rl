# python run.py -m name=rollout_barl_pt_200 alg=rollout_barl num_iters=300 eval_frequency=5 env=plasma_tracking seed="range(5)" hydra/launcher=joblib env.eigmpc.num_iters=2
# python run.py -m name=rollout_barl_pt_800 alg=rollout_barl num_iters=300 eval_frequency=5 env=plasma_tracking seed="range(5)" hydra/launcher=joblib env.eigmpc.num_iters=8
# python run.py -m name=rollout_barl_pt_3200 alg=rollout_barl num_iters=300 eval_frequency=5 env=plasma_tracking seed="range(5)" hydra/launcher=joblib env.eigmpc.num_iters=16 env.eigmpc.nsamps=200
# python run.py -m name=rollout_barl_pt_10000 alg=rollout_barl num_iters=300 eval_frequency=5 env=plasma_tracking seed="range(5)" hydra/launcher=joblib env.eigmpc.num_iters=25 env.eigmpc.nsamps=400
python run.py -m name=barl_pt_200 alg=barl num_iters=300 eval_frequency=5 env=plasma_tracking seed="range(5)" hydra/launcher=joblib alg.n_rand_acqopt=200
python run.py -m name=barl_pt_800 alg=barl num_iters=300 eval_frequency=5 env=plasma_tracking seed="range(5)" hydra/launcher=joblib alg.n_rand_acqopt=800
python run.py -m name=barl_pt_3200 alg=barl num_iters=300 eval_frequency=5 env=plasma_tracking seed="range(5)" hydra/launcher=joblib alg.n_rand_acqopt=3200
python run.py -m name=barl_pt_10000 alg=barl num_iters=300 eval_frequency=5 env=plasma_tracking seed="range(5)" hydra/launcher=joblib alg.n_rand_acqopt=10000
