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
    Modifying the minDQN to implement the Dueling VanillaDQN
    We only need to replace the Q network with the new DuelingNN
    """
    def __init__(self, env, opt, layers,  memory_size=10000, batch_size=100, freq_update_target=100, learning_rate=0.001, explo=0.1, explo_mode=0, discount=0, decay=0.9999, **kwargs):
        super().__init__(env, opt, layers=layers, memory_size=memory_size, batch_size=batch_size, freq_update_target=freq_update_target, learning_rate=learning_rate, explo=explo, explo_mode=explo_mode, discount=discount, decay=decay)

        # Building the network and re-setup the optimizer
        self.Q = DuelingNN(self.featureExtractor.outSize, self.action_space.n, layers, layers)
        self.target_net = deepcopy(self.Q)
        self.optim = torch.optim.Adam(self.Q.parameters(), lr=self.learning_rate)
