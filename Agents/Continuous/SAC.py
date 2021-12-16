from copy import deepcopy

import torch
from torch.distributions import Normal, Uniform

from Agents.Agent import Agent
from Structure.Memory import Memory
from Tools.core import NN


class QNet(torch.nn.Module):
    def __init__(self, in_size, action_space_size, layers, final_activation=None, activation=torch.relu, dropout=0):
        super().__init__()
        self.pre_state   = torch.nn.Linear(in_size, 64)
        self.pre_action  = torch.nn.Linear(action_space_size, 64)
        self.q_net       = NN(128, 1, layers=layers, final_activation=final_activation, activation=activation, dropout=dropout)

    def forward(self, obs, action):
        # independant preprocessing
        p_obs = self.pre_state(obs)
        p_act = self.pre_action(action)

        # Concat
        ipt = torch.cat([p_obs, p_act], dim=-1)
        return self.q_net(ipt)


class PolicyNet(torch.nn.Module):
    def __init__(self, in_size, layers=None, final_activation=None, activation=torch.relu, dropout=0):
        super().__init__()

        if layers is None:
            layers = [30]

        self.encode        = NN(in_size, layers[-1], layers=layers, final_activation=final_activation, activation=activation, dropout=dropout)
        self.softplus      = torch.nn.Softplus()
        self.fc_mu         = torch.nn.Linear(layers[-1], 1)
        self.fc_std        = torch.nn.Linear(layers[-1], 1)

    def forward(self, obs, training=False):
        # Here the policy computes a distribution
        h    = self.encode(obs)
        mu   = self.fc_mu(h)
        std  = self.softplus(self.fc_std(h))

        # Action + first clamping
        dist = Normal(mu, std)
        act  = dist.rsample()
        real_action = torch.tanh(act)

        # Computing the log prob, usefull for the training
        real_log_prob = None
        if training:
            log_prob = dist.log_prob(act)
            real_log_prob = log_prob - torch.log(1 - torch.tanh(act).pow(2) + 1e-7)
        return real_action, real_log_prob


class SAC(Agent):
    def __init__(self, env, opt, layers, batch_per_learn=1, loss='smoothL1', batch_size=64, memory_size=1024, alpha=0.01, alpha_learning_rate=None, **kwargs):
        super().__init__(env, opt )

        self.featureExtractor = opt.featExtractor(env)
        self.loss = torch.nn.SmoothL1Loss() if loss == 'smoothL1' else torch.nn.MSELoss()

        self.p_lr = opt.p_learningRate
        self.q_lr = opt.q_learningRate

        # setup q nets:
        self.q1 = QNet(in_size=self.featureExtractor.outSize, action_space_size=len(self.action_space.low), layers=layers, final_activation=None)
        self.q2 = QNet(in_size=self.featureExtractor.outSize, action_space_size=len(self.action_space.low), layers=layers, final_activation=None)

        self.optim_q1  = torch.optim.Adam(params=self.q1.parameters(), lr=self.q_lr)
        self.optim_q2  = torch.optim.Adam(params=self.q2.parameters(), lr=self.q_lr)

        # setup target q nets:
        self.target_q1 = deepcopy(self.q1)
        self.target_q2 = deepcopy(self.q2)
        self.rho = opt.rho

        # setup policy net:
        self.policy = PolicyNet(self.featureExtractor.outSize, layers, final_activation=None)
        self.optim_policy = torch.optim.Adam(params=self.policy.parameters(), lr=self.p_lr)

        # setup temperature
        self.alpha = torch.Tensor([alpha]).log()
        self.alpha.requires_grad = True
        self.alpha_lr = alpha_learning_rate
        if self.alpha_lr is not None:
            self.optim_alpha = torch.optim.Adam(params=[self.alpha], lr=self.alpha_lr)

        # setup memory
        self.memory = Memory(mem_size=memory_size)
        self.batch_size = batch_size
        self.memory_size = memory_size
        self.batch_per_learn = batch_per_learn

        # setup optim rates
        self.freq_optim = opt.freqOptim
        self.n_events = 0
        self.startEvents = opt.startEvents

        # action clamping boundaries
        self.min = torch.Tensor(self.action_space.low)
        self.max = torch.Tensor(self.action_space.high)

    def store(self, transition):
        if not self.test:
            self.memory.store(transition)

    def act(self, obs):
        if self.n_events < self.startEvents:
            dist = Uniform(self.min, self.max)
            return dist.sample()
        with torch.no_grad():
            a, _ = self.policy(obs)
        return torch.clamp(a, self.min, self.max).view(-1)

    def time_to_learn(self):
        self.n_events += 1
        if self.n_events % self.freq_optim != 0 or self.test:
            return False
        else:
            return True

    def learn(self, done):
        mean_q1loss, mean_q2loss, mean_ploss, mean_aloss = 0, 0, 0, 0
        for k in range(self.batch_per_learn):
            q1loss, q2loss, ploss, aloss = self._train_batch()
            mean_q1loss += q1loss
            mean_q2loss += q2loss
            mean_ploss += ploss
            mean_aloss += aloss
        return {'Q1 Loss': mean_q1loss / self.batch_per_learn, 'Q2 Loss': mean_q2loss / self.batch_per_learn, 'Policy Loss': mean_ploss / self.batch_per_learn, 'Alpha Loss': mean_aloss / self.batch_per_learn}

    def _train_batch(self):
        batches = self.memory.sample_batch(batch_size=self.batch_size)

        b_obs = batches['obs']
        b_action = batches['action'].unsqueeze(-1)
        b_reward = batches['reward']
        b_new = batches['new_obs']
        b_done = batches['done']

        # update q net
        q1_loss, q2_loss = self._update_q(b_obs, b_action, b_reward, b_new, b_done)

        # update policy
        p_loss, alpha_loss = self._update_policy(b_obs, b_action, b_reward, b_new, b_done)

        # update target network
        self._update_target()
        return q1_loss, q2_loss, p_loss, alpha_loss

    def _update_q(self, b_obs, b_action, b_reward, b_new, b_done):
        target = self._compute_objective(b_reward, b_new, b_done)

        # updating q1
        q1_loss = self.loss(self.q1(b_obs, b_action), target)
        self.optim_q1.zero_grad()
        q1_loss.backward()
        self.optim_q1.step()

        # updating q2
        q2_loss = self.loss(self.q1(b_obs, b_action), target)
        self.optim_q2.zero_grad()
        q2_loss.backward()
        self.optim_q2.step()

        return q1_loss.item(), q2_loss.item()

    def _update_policy(self, b_obs, b_action, b_reward, b_new, b_done):
        actions, action_log_prob = self.policy(b_obs, training=True)
        entropy = - self.alpha.exp() * action_log_prob

        q1_val = self.q1(b_obs, actions)
        q2_val = self.q2(b_obs, actions)
        yv = - (torch.min(torch.cat([q1_val, q2_val], dim=1), dim=1, keepdim=True)[0] + entropy).mean()

        self.optim_policy.zero_grad()
        yv.backward()
        self.optim_policy.step()

        alpha_loss = torch.zeros(1)
        if self.alpha_lr is not None:
            self.optim_alpha.zero_grad()
            alpha_loss = -(self.alpha.exp() * (action_log_prob.detach() - 1)).mean()
            alpha_loss.backward()
            self.optim_alpha.step()

        return yv.item(), alpha_loss.item()

    def _update_target(self):
        with torch.no_grad():
            for target_p, net_p in zip(self.target_q1.parameters(), self.q1.parameters()):
                new_p = self.rho * target_p + (1 - self.rho) * net_p
                target_p.copy_(new_p)

            for target_p, net_p in zip(self.target_q2.parameters(), self.q2.parameters()):
                new_p = self.rho * target_p + (1 - self.rho) * net_p
                target_p.copy_(new_p)

    def _compute_objective(self, b_reward, b_new, b_done):
        with torch.no_grad():
            next_action, log_prob = self.policy(b_new, training=True)
            entropy = - self.alpha.exp() * log_prob

            q1_val = self.target_q1(b_new, next_action)
            q2_val = self.target_q2(b_new, next_action)
            v = torch.min(torch.cat([q1_val, q2_val], dim=1), dim=1, keepdim=True)[0] + entropy

        td = b_reward + self.discount * v * (~b_done).float()
        return td
