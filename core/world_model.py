import numpy as np

class WorldModel:
    def __init__(self, latent_dim, action_dim):
        self.latent_dim = latent_dim
        self.action_dim = action_dim

    def predict(self, z, a):
        z = np.array(z)
        a = np.array(a)
        return np.tanh(z + 0.05 * np.pad(a, (0, len(z)-len(a))))
