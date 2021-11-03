from Agents.Agent import Agent
import torch
from Tools.core import NN
from Structure.Memory import Memory
import numpy as np
import numpy.random as random


class A2C(Agent):
    def __init__(self, env, opt, layers, loss='smoothL1', batch_size=10, memory_size=1000):
        super().__init__(env, opt)
        self.featureExtractor = opt.featExtractor(env)
        self.loss = torch.nn.SmoothL1Loss() if loss == 'smoothL1' else torch.nn.MSELoss()
        self.lr = opt.learningRate

        self.policy_net = NN(self.featureExtractor.outSize, self.action_space.n, layers=layers, finalActivation=torch.nn.Sigmoid())
        self.critic_net = NN(self.featureExtractor.outSize, self.action_space.n, layers=layers, finalActivation=torch.nn.Sigmoid())
        self.optim_critic = torch.optim.Adam(self.critic_net.parameters(), lr=self.lr)
        self.optim_policy = torch.optim.Adam(self.policy_net.parameters(), lr=self.lr)

        self.memory = Memory(mem_size=memory_size)
        self.batch_size = batch_size
        self.memory_size = memory_size

        self.lastTransition = {}
        self.freq_optim = self.config.freqOptim
        self.n_events = 0

    def store(self, transition):
        for key, value in transition.items():
            self.lastTransition[key] = value

    def memorize(self):
        self.memory.store(self.lastTransition)

    def act(self, obs):
        with torch.no_grad():
            pick = self.policy_net(obs).reshape(-1)
            self.lastTransition['pi'] = pick

        if random.random_sample() < self.explo:
            return random.randint(self.action_space.n)
        else:
            return np.argmax(pick).item()

    def time_to_learn(self):
        if self.memory.nentities < self.memory_size or self.test:
            return False
        else:
            self.n_events += 1
            if self.n_events % self.freq_optim == 0:
                return True
            return False

    def learn(self, done):

        batches = self.memory.sample_batch(n_batch=self.memory_size // self.batch_size, batch_size=self.batch_size)

        b_obs = batches['obs']
        b_action = batches['action']
        b_reward = batches['reward']
        b_new = batches['new_state']
        b_done = batches['done']
        b_pi = batches['pi']

        j = torch.zeros(1, requires_grad=True)
        train_loss = 0
        for i in range(self.memory_size // self.batch_size):
            obs, action, reward, new, done, pi = b_obs[i], b_action[i], b_reward[i], b_new[i], b_done[i], b_pi[i]

            qhat = self.critic_net(new)
            with torch.no_grad():
                r = (reward + self.discount * torch.max(qhat, dim=-1).values * (~done)).float()

            loss = self.loss(r, torch.gather(qhat, -1, action.reshape(-1, 1).long()).squeeze())
            loss.backward()
            train_loss += loss.item()

            self.optim_critic.step()
            self.optim_critic.zero_grad()

            with torch.no_grad():
                advantage = qhat * pi
            for policy_sample, ad in zip(pi, advantage):
                j = j + torch.log(policy_sample).T @ ad

        inv_j = -1 * j
        inv_j.backward()
        self.optim_policy.step()
        self.optim_policy.zero_grad()

        # reset memory for next training
        del self.memory
        self.memory = Memory(mem_size=self.memory_size)

        return train_loss
