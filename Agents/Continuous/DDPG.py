from copy import deepcopy

import torch

from Agents.Agent import Agent
from Structure.Memory import Memory
from Tools.core import NN
from Tools.core import Orn_Uhlen


class QPolicyNet:
    def __init__(self, in_size, action_space_size, layers, final_activation=None, activation=torch.relu, dropout=0):

        self.q_net      = NN(in_size + action_space_size, 1, layers=layers, final_activation=None, activation=activation, dropout=dropout)
        self.policy_net = NN(in_size, action_space_size, layers=layers, final_activation=final_activation, activation=activation, dropout=dropout)

    def policy(self, x):
        x = x.float()
        return self.policy_net(x)

    def q(self, obs, action):
        ipt = torch.cat([obs, action], dim=-1)
        return self.q_net(ipt)


class DDPG(Agent):
    def __init__(self, env, opt, layers, batch_per_learn=10, loss='smoothL1', batch_size=1000, memory_size=10000, **kwargs):
        super().__init__(env, opt)

        self.featureExtractor = opt.featExtractor(env)
        self.loss = torch.nn.SmoothL1Loss() if loss == 'smoothL1' else torch.nn.MSELoss()
        self.p_lr = opt.p_learningRate
        self.q_lr = opt.q_learningRate

        self.network = QPolicyNet(in_size=self.featureExtractor.outSize, action_space_size=len(self.action_space.low), layers=layers, final_activation=torch.nn.Tanh())
        self.target_net = deepcopy(self.network)
        self.optim_q      = torch.optim.Adam(params=self.network.q_net.parameters(), lr=self.q_lr)
        self.optim_policy = torch.optim.Adam(params=self.network.policy_net.parameters(), lr=self.p_lr)

        self.memory = Memory(mem_size=memory_size)
        self.batch_size = batch_size
        self.memory_size = memory_size
        self.batch_per_learn = batch_per_learn

        self.noise = Orn_Uhlen(n_actions=len(self.action_space.low))  # ??

        self.rho = opt.rho

        self.freq_optim = self.config.freqOptim
        self.n_events   = 0

        self.min = torch.Tensor(self.action_space.low)
        self.max = torch.Tensor(self.action_space.high)

    def store(self, transition):
        if not self.test:
            self.memory.store(transition)

    def act(self, obs):
        with torch.no_grad():
            a = self.network.policy(obs)
        n = torch.zeros(1)
        if not self.test:
            n = self.noise.sample()
            # n = torch.rand(1) * 0.1
        o = (a+n).squeeze()
        return torch.clamp(o, self.min, self.max)

    def time_to_learn(self):
        self.n_events += 1
        if self.n_events % self.freq_optim != 0 or self.test:
            return False
        else:
            return True

    def learn(self, episode_done):
        mean_qloss, mean_ploss = 0, 0
        for k in range(self.batch_per_learn):
            qloss, ploss = self._train_batch()
            mean_qloss += qloss
            mean_ploss += ploss
        return {'Q Loss': mean_qloss/self.batch_per_learn, 'Policy Loss': mean_ploss/self.batch_per_learn}

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
        loss_policy = self._update_policy(b_obs)

        # update target network
        self._update_target()

        return q_loss.item(), loss_policy.item()

    def _update_q(self, b_obs, b_action, b_reward, b_new, b_done):

        # on commence par calculer la target
        with torch.no_grad():
            target_next_act = self.target_net.policy(b_new)
            target = b_reward + self.discount * self.target_net.q(b_new, target_next_act).squeeze() * (~b_done).float()

        preds_q = self.network.q(b_obs, b_action)
        q_loss = self.loss(preds_q, target)

        self.optim_q.zero_grad()
        q_loss.backward()
        self.optim_q.step()

        return q_loss

    def _update_policy(self, b_obs):
        pred_act = self.network.policy(b_obs)
        pred_q = self.network.q(b_obs, pred_act)
        loss_policy = -torch.mean(pred_q)

        self.optim_policy.zero_grad()
        loss_policy.backward()
        self.optim_policy.step()

        return loss_policy

    def _update_target(self):
        with torch.no_grad():
            for target_p, net_p in zip(self.target_net.policy_net.parameters(), self.network.policy_net.parameters()):
                new_p = self.rho * target_p + (1 - self.rho) * net_p
                target_p.copy_(new_p)

            for target_p, net_p in zip(self.target_net.q_net.parameters(), self.network.q_net.parameters()):
                new_p = self.rho * target_p + (1 - self.rho) * net_p
                target_p.copy_(new_p)




