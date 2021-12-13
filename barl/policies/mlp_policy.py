from tensorflow import keras
from tensorflow.keras import layers
keras.backend.set_floatx('float64')


class MlpPolicy:
    def __init__(self, obs_dim, action_dim, hidden_layer_sizes, output_activation=None):
        policy_layers = [layers.Dense(hidden_layer_sizes[0], activation="relu", input_shape=(obs_dim,))] +\
                        [layers.Dense(size, activation="relu") for size in hidden_layer_sizes[1:]]
        policy_layers.append(layers.Dense(action_dim, activation=output_activation))
        self.model = keras.Sequential(policy_layers)

    def __call__(self, obs):
        return self.model(obs)

    def save(self, path):
        self.model.save(path)

    def load(self, path):
        self.model = keras.models.load_model(path)


class TanhMlpPolicy(MlpPolicy):
    def __init__(self, obs_dim, action_dim, hidden_layer_sizes):
        super().__init__(obs_dim, action_dim, hidden_layer_sizes, output_activation='tanh')
