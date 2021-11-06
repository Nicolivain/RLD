from Agents.Agent import Agent
import torch
from Tools.core import NN
from Structure.Memory import Memory
import numpy as np
import numpy.random as random
from Tools.exploration import *


class A2C(Agent):
    def __init__(self, env, opt, layers, loss='smoothL1', batch_size=10, memory_size=1000):
        super().__init__(env, opt)
        self.featureExtractor = opt.featExtractor(env)
        self.loss = torch.nn.SmoothL1Loss() if loss == 'smoothL1' else torch.nn.MSELoss()
        self.lr = opt.learningRate

        self.policy_net = NN(self.featureExtractor.outSize, self.action_space.n, layers=layers, finalActivation=torch.nn.Softmax())
        self.critic_net = NN(self.featureExtractor.outSize, 1, layers=layers, finalActivation=torch.nn.Sigmoid())
        self.optim_critic = torch.optim.Adam(self.critic_net.parameters(), lr=self.lr)
        self.optim_policy = torch.optim.Adam(self.policy_net.parameters(), lr=self.lr)

        self.memory = Memory(mem_size=memory_size)
        self.batch_size = batch_size
        self.memory_size = memory_size

        self.lastTransition = {}
        self.freq_optim = self.config.freqOptim
        self.n_events = 0

    def store(self, transition):
        if not self.test:
            self.memory.store(transition)

    def memorize(self):
        self.memory.store(self.lastTransition)

    def act(self, obs):
        with torch.no_grad():
            values = self.policy_net(obs).reshape(-1)
        if self.test:
            return pick_greedy(values.numpy())
        else:
            return pick_sample(values)

    def time_to_learn(self):
        if self.memory.nentities < self.memory_size or self.test:
            return False
        else:
            self.n_events += 1
            if self.n_events % self.freq_optim == 0:
                return True
            return False

    def learn(self, done):

        batches = self.memory.sample_batch(batch_size=self.batch_size)

        b_obs    = batches['obs']
        b_action = batches['action']
        b_reward = batches['reward']
        b_new    = batches['new_state']
        b_done   = batches['done']

        pi = self.policy_net(b_obs)
        critic = self.critic_net(b_obs).squeeze()
        with torch.no_grad():
            next_critic = self.critic_net(b_new)
            td0 = (b_reward + self.discount * next_critic.squeeze() * (~b_done)).float()
            advantage = td0 - critic

        loss = self.loss(td0, critic)
        action_pi = pi.gather(-1, b_action.reshape(-1, 1).long()).squeeze()
        j = -torch.mean(torch.log(action_pi) * advantage)

        loss.backward()
        self.optim_critic.step()
        self.optim_critic.zero_grad()

        j.backward()
        self.optim_policy.step()
        self.optim_policy.zero_grad()

        # reset memory for next training
        del self.memory
        self.memory = Memory(mem_size=self.memory_size)

        return j.item()
