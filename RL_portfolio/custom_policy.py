import torch as th
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class CustomPolicyNetwork(BaseFeaturesExtractor):
    """
    Rete neurale personalizzata per estrarre feature dallo stato.
    """
    def __init__(self, observation_space, features_dim=64):
        super(CustomPolicyNetwork, self).__init__(observation_space, features_dim)

        # Calcola il numero di input dallo spazio degli stati
        input_dim = observation_space.shape[0]

        # Definisci la rete neurale
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, features_dim),
            nn.ReLU()
        )

    def forward(self, observations):
        return self.net(observations)
