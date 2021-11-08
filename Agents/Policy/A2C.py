import torch

from Agents.Agent import Agent
from Structure.Memory import Memory
from Tools.core import NN
from Tools.exploration import *


class ACNet(torch.nn.Module):
    # TODO test simpler architecture ie One layer
    def __init__(self, in_size, action_space_size, layers, final_activation=None, activation=torch.relu, dropout=0):
        super().__init__()
        self.policy_net = NN(in_size, action_space_size, layers=layers, finalActivation=final_activation, activation=activation, dropout=dropout)
        self.critic_net = NN(in_size, 1, layers=layers, finalActivation=None, activation=activation, dropout=dropout)

    def policy(self, x):
        return self.policy_net(x)

    def critic(self, x):
        return self.critic_net(x)


class A2C(Agent):
    def __init__(self, env, opt, layers, loss='smoothL1', batch_size=10, memory_size=1000):
        super().__init__(env, opt)
        self.featureExtractor = opt.featExtractor(env)
        self.loss = torch.nn.SmoothL1Loss() if loss == 'smoothL1' else torch.nn.MSELoss()
        self.lr = opt.learningRate

        self.model = ACNet(self.featureExtractor.outSize, self.action_space.n, layers, final_activation=torch.nn.Softmax(dim=-1))
        self.optim = torch.optim.Adam(params=self.model.parameters(), lr=self.lr)

        self.memory = Memory(mem_size=memory_size)
        self.batch_size = batch_size
        self.memory_size = memory_size

        self.n_events = 0

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

    def learn(self, done):

        batches = self.memory.sample_batch(batch_size=self.batch_size)

        b_obs    = batches['obs']
        b_action = batches['action']
        b_reward = batches['reward']
        b_new    = batches['new_obs']
        b_done   = batches['done']

        pi = self.model.policy(b_obs)
        critic = self.model.critic(b_obs).squeeze()
        with torch.no_grad():
            next_critic = self.model.critic(b_new)
            td0 = (b_reward + self.discount * next_critic.squeeze() * (~b_done)).float()
            advantage = td0 - critic

        loss = self.loss(td0, critic)
        action_pi = pi.gather(-1, b_action.reshape(-1, 1).long()).squeeze()
        j = -torch.mean(torch.log(action_pi) * advantage)

        self.optim.zero_grad()
        loss.backward()
        j.backward()
        self.optim.step()

        # reset memory for next training
        del self.memory
        self.memory = Memory(mem_size=self.memory_size)

        return loss.item()
