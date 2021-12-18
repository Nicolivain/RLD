import gym
import numpy.random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
import collections, random

from Structure.Memory import Memory
from Agents.Agent import Agent

# Hyperparameters
lr_pi = 0.0005
lr_q = 0.001
init_alpha = 0.01
gamma = 0.98
batch_size = 32
buffer_limit = 50000
tau = 0.01  # for target network soft update
target_entropy = -1.0  # for automated alpha update
lr_alpha = 0.001  # for automated alpha update


class PolicyNet(nn.Module):
    def __init__(self):
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(3, 128)
        self.fc_mu = nn.Linear(128, 1)
        self.fc_std = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        mu = self.fc_mu(x)
        std = F.softplus(self.fc_std(x))
        dist = Normal(mu, std)
        action = dist.rsample()
        log_prob = dist.log_prob(action)
        real_action = torch.tanh(action)
        real_log_prob = log_prob - torch.log(1 - torch.tanh(action).pow(2) + 1e-7)
        return real_action, real_log_prob

    def train_net(self, q1, q2, mini_batch):
        s, _, _, _, _ = mini_batch
        a, log_prob = self.forward(s)
        entropy = -self.log_alpha.exp() * log_prob

        q1_val, q2_val = q1(s, a), q2(s, a)
        q1_q2 = torch.cat([q1_val, q2_val], dim=1)
        min_q = torch.min(q1_q2, 1, keepdim=True)[0]

        loss = -min_q - entropy  # for gradient ascent
        self.optimizer.zero_grad()
        loss.mean().backward()
        self.optimizer.step()

        self.log_alpha_optimizer.zero_grad()
        alpha_loss = -(self.log_alpha.exp() * (log_prob + target_entropy).detach()).mean()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()


class QNet(nn.Module):
    def __init__(self):
        super(QNet, self).__init__()
        self.fc_s = nn.Linear(3, 64)
        self.fc_a = nn.Linear(1, 64)
        self.fc_cat = nn.Linear(128, 32)
        self.fc_out = nn.Linear(32, 1)

    def forward(self, x, a):
        h1 = F.relu(self.fc_s(x))
        h2 = F.relu(self.fc_a(a))
        cat = torch.cat([h1, h2], dim=1)
        q = F.relu(self.fc_cat(cat))
        q = self.fc_out(q)
        return q


class SAC(Agent):
    def __init__(self, env, opt):
        super().__init__(env, opt)

        self.memory = Memory(buffer_limit)

        self.q1 = QNet()
        self.q2 = QNet()
        self.optim_q1 = optim.Adam(self.q1.parameters(), lr=lr_q)
        self.optim_q2 = optim.Adam(self.q2.parameters(), lr=lr_q)

        self.q1_target = QNet()
        self.q2_target = QNet()

        self.policy = PolicyNet()
        self.optim_policy = optim.Adam(self.policy.parameters(), lr=lr_pi)

        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        self.log_alpha = torch.tensor(np.log(init_alpha))
        self.log_alpha.requires_grad = True
        self.log_alpha_optimizer = optim.Adam([self.log_alpha], lr=lr_alpha)

    def act(self, obs):
        a, _ = self.policy(obs)
        return a.item()

    def learn(self, done):
        for i in range(20):
            mini_batch = self.memory.sample_batch(batch_size)

            obs     = mini_batch['obs']
            action  = mini_batch['action']
            reward  = mini_batch['reward']
            new_obs = mini_batch['new_obs']
            done    = mini_batch['done']

            t = (obs, action, reward, new_obs, done)
            td_target = self._compute_objective(obs, action, reward, new_obs, done)
            self._update_qnet(td_target, obs, action)
            self._update_policy(obs)

            self._update_target()

    def time_to_learn(self, done):
        if done and self.memory.nentities > 1000:
            return True
        else:
            return False

    def store(self, transition):
        self.memory.put(transition)

    def _compute_objective(self, obs, action, reward, new_obs, done):
        with torch.no_grad():
            a_prime, log_prob = self.policy(new_obs)
            entropy = -self.log_alpha.exp() * log_prob
            q1_val, q2_val = self.q1_target(new_obs, a_prime), self.q2_target(new_obs, a_prime)

            q1_q2 = torch.cat([q1_val, q2_val], dim=1)
            min_q = torch.min(q1_q2, 1, keepdim=True)[0]
            target = reward + gamma * done * (min_q + entropy)

        return target

    def _update_qnet(self, target, obs, action):
        loss = F.smooth_l1_loss(self.q1(obs, action), target)
        self.optim_q1.zero_grad()
        loss.mean().backward()
        self.optim_q1.step()

        loss = F.smooth_l1_loss(self.q2(obs, action), target)
        self.optim_q2.zero_grad()
        loss.mean().backward()
        self.optim_q2.step()

    def _update_policy(self, s):
        a, log_prob = self.policy(s)
        entropy = -self.log_alpha.exp() * log_prob

        q1_val, q2_val = self.q1(s, a), self.q2(s, a)
        q1_q2 = torch.cat([q1_val, q2_val], dim=1)
        min_q = torch.min(q1_q2, 1, keepdim=True)[0]

        loss = -min_q - entropy  # for gradient ascent
        self.optim_policy.zero_grad()
        loss.mean().backward()
        self.optim_policy.step()

        self.log_alpha_optimizer.zero_grad()
        alpha_loss = -(self.log_alpha.exp() * (log_prob + target_entropy).detach()).mean()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

    def _update_target(self):
        for param_target, param in zip(self.q1_target.parameters(), self.q1.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - tau) + param.data * tau)

        for param_target, param in zip(self.q2_target.parameters(), self.q2.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - tau) + param.data * tau)

