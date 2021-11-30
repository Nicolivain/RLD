from copy import deepcopy

import torch
from torch.distributions import Normal

from Agents.Agent import Agent
from Structure.Memory import Memory
from Tools.core import NN


class QNet(torch.nn.Module):
    def __init__(self, in_size, action_space_size, layers, final_activation=None, activation=torch.relu, dropout=0):
        super().__init__()
        self.q_net      = NN(in_size + action_space_size, 1, layers=layers, final_activation=final_activation, activation=activation, dropout=dropout)

    def forward(self, obs, action):
        ipt = torch.cat([obs, action], dim=-1)
        return self.q_net(ipt)

class PolicyNet(torch.nn.Module):
    def __init__(self, in_size, layers, final_activation=None, activation=torch.relu, dropout=0):
        super().__init__()

        if not layers:
            layers = [30]

        self.encode        = NN(in_size, layers[-1], layers=layers, final_activation=final_activation, activation=activation, dropout=dropout)
        self.softplus      = torch.nn.Softplus()
        self.fc_mu         = torch.nn.Linear(layers[-1], 1)
        self.fc_std        = torch.nn.Linear(layers[-1], 1)

    def forward(self, obs):
        h    = self.encode(obs)
        mu   = self.fc_mu(h)
        std  = self.softplus(self.fc_std(h))
        dist = Normal(mu, std)
        act  = dist.rsample()

        # TODO: understand this (cf minimal RL)
        log_prob = dist.log_prob(act)
        real_action = torch.tanh(act)
        real_log_prob = log_prob - torch.log(1 - torch.tanh(act).pow(2) + 1e-7)
        return real_action, real_log_prob


class SAC(Agent):
    def __init__(self, env, opt, layers, batch_per_learn=1, loss='smoothL1', batch_size=64, memory_size=1024):
        super().__init__(env, opt )

        self.featureExtractor = opt.featExtractor(env)
        self.loss = torch.nn.SmoothL1Loss() if loss == 'smoothL1' else torch.nn.MSELoss()
        self.p_lr = opt.p_learningRate
        self.q_lr = opt.q_learningRate

        # TODO: not sure about the networks
        # setup q nets:
        self.q1 = QNet(in_size=self.featureExtractor.outSize, action_space_size=len(self.action_space.low), layers=layers, final_activation=None)
        self.q2 = QNet(in_size=self.featureExtractor.outSize, action_space_size=len(self.action_space.low), layers=layers, final_activation=None)

        self.optim_q1  = torch.optim.Adam(params=self.q1.parameters(), lr=self.q_lr)
        self.optim_q2  = torch.optim.Adam(params=self.q2.parameters(), lr=self.q_lr)

        # setup target q nets:
        self.target_q1 = deepcopy(self.q1)
        self.target_q2 = deepcopy(self.q2)

        # setup policy net:
        self.policy = PolicyNet(self.featureExtractor.outSize, layers, final_activation=None)
        self.optim_policy = torch.optim.Adam(params=self.policy.parameters(), lr=self.p_lr)

        self.memory = Memory(mem_size=memory_size)
        self.batch_size = batch_size
        self.memory_size = memory_size
        self.batch_per_learn = batch_per_learn

        self.rho = opt.rho

        self.freq_optim = self.config.freqOptim
        self.n_events = 0

        self.min = torch.Tensor(self.action_space.low)
        self.max = torch.Tensor(self.action_space.high)

    def store(self, transition):
        if not self.test:
            self.memory.store(transition)

    def act(self, obs):
        with torch.no_grad():
            a, _ = self.policy(obs)
        return torch.clamp(a, self.min, self.max)

    def time_to_learn(self):
        self.n_events += 1
        if self.n_events % self.freq_optim != 0 or self.test:
            return False
        else:
            return True

    def _train_batch(self):
        batches = self.memory.sample_batch(batch_size=self.batch_size)

        b_obs = batches['obs']
        b_action = batches['action'].unsqueeze(-1)
        b_reward = batches['reward']
        b_new = batches['new_obs']
        b_done = batches['done']

        # update q net
        q_loss = self._update_q(b_obs, b_action, b_reward, b_new, b_done)

        # update policy
        loss_policy = self._update_policy((b_obs, b_action, b_reward, b_new, b_done)

        # update target network
        self._update_target()

    def _update_q(self, b_obs, b_action, b_reward, b_new, b_done):
        # TODO: how to compute targets ?
        with torch.no_grad():
            target_next_act = self.target_q1.policy(b_new)
            yq = b_reward + self.discount * self.target_q1.q(b_new, target_next_act).squeeze() * (~b_done).float()



    def _update_policy(self, b_obs, b_action, b_reward, b_new, b_done):
        # TODO: understand this
        pass


    def _update_target(self):
        with torch.no_grad():
            for target_p, net_p in zip(self.target_q1.parameters(), self.q1.parameters()):
                new_p = self.rho * target_p + (1 - self.rho) * net_p
                target_p.copy_(new_p)

            for target_p, net_p in zip(self.target_q2.parameters(), self.q2.parameters()):
                new_p = self.rho * target_p + (1 - self.rho) * net_p
                target_p.copy_(new_p)
