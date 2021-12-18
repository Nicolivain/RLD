import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import numpy as np

from Structure.Memory import Memory
from Agents.Agent import Agent


class PolicyNet(nn.Module):
    def __init__(self, input_size):
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
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


class QNet(nn.Module):
    def __init__(self, state_input_size):
        super(QNet, self).__init__()
        self.fc_s = nn.Linear(state_input_size, 64)
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
    def __init__(self, env, opt, batch_size=32, memory_size=50000, batch_per_learn=20, init_alpha=0.01, target_entropy=-1, lr_alpha=0.001, tune_alpha=True, **kwargs):
        super().__init__(env, opt)

        # parameters
        self.batch_size         = batch_size
        self.memory_size        = memory_size
        self.batch_per_learn    = batch_per_learn
        self.target_entropy     = target_entropy

        self.lr_policy = opt.p_learningRate
        self.lr_q      = opt.q_learningRate
        self.lr_alpha  = lr_alpha
        self.rho       = opt.rho

        # setup memory
        self.memory = Memory(self.memory_size)

        # setup Q networks
        self.q1 = QNet(state_input_size=self.featureExtractor.outSize)
        self.q2 = QNet(state_input_size=self.featureExtractor.outSize)
        self.optim_q1 = optim.Adam(self.q1.parameters(), lr=self.lr_q)
        self.optim_q2 = optim.Adam(self.q2.parameters(), lr=self.lr_q)

        # setup Q target networks
        self.q1_target = QNet(state_input_size=self.featureExtractor.outSize)
        self.q2_target = QNet(state_input_size=self.featureExtractor.outSize)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        # setup Policy network
        self.policy = PolicyNet(input_size=self.featureExtractor.outSize)
        self.optim_policy = optim.Adam(self.policy.parameters(), lr=self.lr_policy)

        # setup alpha tuning
        self.tune_alpha = tune_alpha
        self.log_alpha  = torch.tensor(np.log(init_alpha))
        self.log_alpha.requires_grad = True
        self.log_alpha_optimizer = optim.Adam([self.log_alpha], lr=self.lr_alpha)

    def act(self, obs):
        a, _ = self.policy(obs)
        return a.item()

    def learn(self, done):
        res = {'Policy Loss': 0, 'Q1 Loss': 0, 'Q2 Loss': 0, 'Entropy': 0, 'Alpha Loss': 0}
        for i in range(self.batch_per_learn):
            mini_batch = self.memory.sample_batch(self.batch_size)

            obs     = mini_batch['obs']
            action  = mini_batch['action']
            reward  = mini_batch['reward']
            new_obs = mini_batch['new_obs']
            done    = mini_batch['done']

            td_target               = self._compute_objective(reward, new_obs, done)
            lq1, lq2                = self._update_qnet(td_target, obs, action)
            lp, alpha_loss, entropy = self._update_policy(obs)
            self._update_target()

            res['Policy Loss']  += lp
            res['Q1 Loss']      += lq1
            res['Q2 Loss']      += lq2
            res['Entropy']      += entropy
            res['Alpha Loss']   += alpha_loss

        res = {k: v/self.batch_per_learn for k, v in res.items()}
        res['Alpha'] = self.log_alpha.exp()
        return res

    def time_to_learn(self):
        if self.memory.nentities > 1000:
            return True
        else:
            return False

    def store(self, transition):
        self.memory.store(transition)

    def _compute_objective(self, reward, new_obs, done):
        with torch.no_grad():
            a_prime, log_prob = self.policy(new_obs)
            entropy = -self.log_alpha.exp() * log_prob
            q1_val, q2_val = self.q1_target(new_obs, a_prime), self.q2_target(new_obs, a_prime)

            q1_q2 = torch.cat([q1_val, q2_val], dim=1)
            min_q = torch.min(q1_q2, 1, keepdim=True)[0]
            target = reward + self.discount * (~done).float() * (min_q + entropy)

        return target

    def _update_qnet(self, target, obs, action):
        lossq1 = F.smooth_l1_loss(self.q1(obs, action), target)
        self.optim_q1.zero_grad()
        lossq1.mean().backward()
        self.optim_q1.step()

        lossq2 = F.smooth_l1_loss(self.q2(obs, action), target)
        self.optim_q2.zero_grad()
        lossq2.mean().backward()
        self.optim_q2.step()
        return lossq1.mean().item(), lossq2.mean().item()

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

        alpha_loss = 0
        if self.tune_alpha:
            self.log_alpha_optimizer.zero_grad()
            alpha_loss = -(self.log_alpha.exp() * (log_prob + self.target_entropy).detach()).mean()
            alpha_loss.backward()
        self.log_alpha_optimizer.step()
        return loss.mean().item(), alpha_loss.item(), entropy.mean().item()

    def _update_target(self):
        for param_target, param in zip(self.q1_target.parameters(), self.q1.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.rho) + param.data * self.rho)

        for param_target, param in zip(self.q2_target.parameters(), self.q2.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.rho) + param.data * self.rho)

