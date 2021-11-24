from copy import deepcopy

import torch

from Agents.DQN.DQN import DQN
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


class DuelingDQN(DQN):
    def __init__(self, env, config, layers, loss='mse', memory_size=10000, batch_size=100, update_target=100):
        super().__init__(env, config, layers, loss, memory_size)
        self.batch_size = batch_size

        # Building the network and re-setup the optimizer
        self.Q = DuelingNN(self.featureExtractor.outSize, self.action_space.n, layers, layers)
        self.target_net = deepcopy(self.Q)
        self.optim = torch.optim.Adam(self.Q.parameters(), lr=self.alpha)

        self.update_target = update_target
        self.n_learn = 0

    def time_to_learn(self):
        if self.memory.nentities < self.memory_size:
            return False
        else:
            return super().time_to_learn()

    def learn(self, episode_done):
        batches = self.memory.sample_batch(batch_size=self.batch_size)
        b_obs = batches['obs']
        b_action = batches['action']
        b_reward = batches['reward'].float()
        b_next = batches['new_obs']
        b_done = batches['done']

        qhat = self.Q(b_obs)  # we use the adjusted Q instead of the vanilla Q-net
        learning = torch.gather(qhat, -1, b_action.reshape(-1, 1)).squeeze()

        with torch.no_grad():
            qhat_target = self.target_net(b_next)
            r = b_reward + self.discount * torch.max(qhat_target, dim=-1).values * (1 - b_done.float())

        loss = self.loss(learning, r)
        loss.backward()

        self.optim.step()
        self.optim.zero_grad()

        self.n_learn += 1
        if self.n_learn % self.update_target == 0:
            self.target_net.load_state_dict(self.Q.state_dict())

        if episode_done:
            self.explo *= self.decay

        return loss.item()
