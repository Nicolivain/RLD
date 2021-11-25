from copy import deepcopy

import torch

from Agents.DQN.minDQN import MinDQN
from Tools.core import NN


class DuelingNN(torch.nn.Module):
    def __init__(self, obs_dim, action_space_dim, advantage_layers, value_layers):
        super(DuelingNN, self).__init__()

        self.action_space_dim = action_space_dim

        self.value = NN(obs_dim, 1, value_layers)
        self.advantage = NN(obs_dim, action_space_dim, advantage_layers)

    def forward(self, obs):
        v = self.value(obs)
        a = self.advantage(obs)
        return v + a - a.sum(dim=-1).reshape(-1, 1) / self.action_space_dim


class DuelingDQN(MinDQN):
    """
    Modifying the minDQN to implement the Dueling DQN
    We only need to replace the Q network with the new DuelingNN
    """
    def __init__(self, env, config, layers, loss='mse', memory_size=10000, batch_size=100, update_target=100):
        super().__init__(env, config, layers, loss, memory_size, batch_size, update_target)

        # Building the network and re-setup the optimizer
        self.Q = DuelingNN(self.featureExtractor.outSize, self.action_space.n, layers, layers)
        self.target_net = deepcopy(self.Q)
        self.optim = torch.optim.Adam(self.Q.parameters(), lr=self.alpha)
