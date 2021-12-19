import torch

from Agents.Agent import Agent
from Structure.Memory import Memory
from Tools.distributions import batched_dkl
from Tools.core import NN
import torch.nn as nn
import torch.nn.functional as F
from Tools.exploration import *


class PPONetwork(nn.Module):
    def __init__(self, in_size, action_space_size, layers, final_activation_v=None, final_activation_p=torch.nn.Sigmoid(), activation_v=torch.nn.LeakyReLU(), activation_p=torch.nn.LeakyReLU(), dropout=0):
        super().__init__()

        self.value_net  = NN(in_size, 1, layers=layers, final_activation=final_activation_v, activation=activation_v, dropout=dropout)
        self.policy_net = NN(in_size, action_space_size, layers=layers, final_activation=final_activation_p, activation=activation_p, dropout=dropout)

    def policy(self, x):
        x = x.float()
        return self.policy_net(x)

    def value(self, obs):
        obs = obs.float()
        return self.value_net(obs)


class AdaptativePPO(Agent):
    def __init__(self, env, opt, layers, k, delta=1e-3, memory_size=1000, batch_size=1000, use_dkl=True, reversed_dkl=False, learning_rate=0.001, discount=0.99, **kwargs):
        super().__init__(env, opt)

        self.featureExtractor = opt.featExtractor(env)
        self.loss = torch.nn.SmoothL1Loss()
        self.lr = learning_rate
        self.discount = discount

        self.beta = 1
        self.delta = delta
        self.k = k
        self.min_beta = 1e-5

        self.dkl = use_dkl
        self.reversed = reversed_dkl

        self.model = PPONetwork(self.featureExtractor.outSize, self.action_space.n, layers)
        self.optim = torch.optim.Adam(params=self.model.parameters(), lr=self.lr)

        self.memory = Memory(mem_size=memory_size)
        self.batch_size = batch_size if batch_size else memory_size
        self.memory_size = memory_size

    def store(self, transition):
        if not self.test:
            self.memory.store(transition)

    def act(self, obs):
        with torch.no_grad():
            values = self.model.policy(obs).reshape(-1)
        return pick_sample(values)

    def time_to_learn(self):
        if self.memory.nentities < self.memory_size or self.test:
            return False
        else:
            return True

    def _update_betas(self, obs, old_pi):
        new_pi = self.model.policy(obs)
        dkl = batched_dkl(new_pi, old_pi)

        if dkl >= 1.5 * self.delta:
            self.beta *= 2
        elif dkl <= self.delta / 1.5:
            self.beta /= 2

        # clipping the value in case we go to low
        if self.beta < self.min_beta:
            self.beta = self.min_beta

    def _update_value_network(self, obs, reward, next_obs, done):
        with torch.no_grad():
            td0 = (reward + self.discount * self.model.value(next_obs) * (~done)).float()
        loss = self.loss(td0, self.model.value(obs))
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        return loss.item()

    def _compute_objective(self, advantage, pi, new_pi, new_action_pi, action_pi):
        # compute l_theta_theta_k
        advantage_loss = torch.mean(advantage * new_action_pi / action_pi)

        # compute DKL_theta/theta_k
        dkl = 0
        if self.dkl:
            if self.reversed:
                # if we want to experiment with the reversed dkl
                dkl = batched_dkl(pi, new_pi)
            else:
                dkl = batched_dkl(new_pi, pi)

        # computing the adjusted loss
        return -(advantage_loss - self.beta * dkl)

    def learn(self, done):
        batches = self.memory.sample_batch(batch_size=self.batch_size)

        bs          = batches['obs'].shape[0]
        b_obs       = batches['obs'].view(bs, -1)
        b_action    = batches['action'].view(bs, -1)
        b_reward    = batches['reward'].view(bs, -1)
        b_new       = batches['new_obs'].view(bs, -1)
        b_done      = batches['done'].view(bs, -1)

        with torch.no_grad():
            # compute policy and value
            pi = self.model.policy(b_obs)
            values = self.model.value(b_obs)

            # compute td0
            next_value = self.model.value(b_new)
            td0 = (b_reward + self.discount * next_value * (~b_done)).float()

            # compute advantage and action_probabilities
            advantage = td0 - values
            action_pi = pi.gather(-1, b_action.reshape(-1, 1).long())

        avg_policy_loss = 0
        for i in range(self.k):
            # get the new action probabilities
            new_pi = self.model.policy(b_obs)
            new_action_pi = new_pi.gather(-1, b_action.reshape(-1, 1).long())

            # compute the objective with the new probabilities
            objective = self._compute_objective(advantage, pi, new_pi, new_action_pi, action_pi)
            avg_policy_loss += objective.item()

            # optimize
            self.optim.zero_grad()
            objective.backward()
            self.optim.step()

        # Updating betas
        self._update_betas(b_obs, pi)

        # updating value network just like in A2C
        loss = self._update_value_network(b_obs, b_reward, b_new, b_done)

        #  reset memory
        del self.memory
        self.memory = Memory(mem_size=self.memory_size)

        return {'Avg Policy Loss': avg_policy_loss/self.k, 'Value Loss': loss, 'Beta': self.beta}
