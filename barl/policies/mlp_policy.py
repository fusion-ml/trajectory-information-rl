from tensorflow import keras
from tensorflow.keras import layers


def MlpPolicy(obs_dim, action_dim, hidden_layer_sizes, output_activation=None):
    policy_layers = [
        layers.Dense(hidden_layer_sizes[0], activation="relu", input_shape=(obs_dim,))
    ] + [layers.Dense(size, activation="relu") for size in hidden_layer_sizes[1:]]
    policy_layers.append(layers.Dense(action_dim, activation=output_activation))
    return keras.Sequential(policy_layers)


def TanhMlpPolicy(obs_dim, action_dim, hidden_layer_sizes):
    return MlpPolicy(obs_dim, action_dim, hidden_layer_sizes, output_activation="tanh")
