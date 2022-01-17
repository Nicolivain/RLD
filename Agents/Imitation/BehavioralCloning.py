import pickle

import torch
from Tools.core import NN
from Agents.Agent import Agent
from Tools.exploration import *


class BehavioralCloning(Agent):
    def __init__(self, env, opt, expert_data_path, learning_rate=0.001, discount=0.99, batch_size=100, **kwargs):
        super(BehavioralCloning, self).__init__(env, opt)

        self.featureExtractor = opt.featExtractor(env)
        self.loss = torch.nn.SmoothL1Loss()
        self.lr = learning_rate
        self.discount = discount
        self.batch_size = batch_size

        self.expert_states, self.expert_actions = self.load_expert_transition(expert_data_path)

        self.model = NN(self.featureExtractor.outSize, self.action_space.n, layers=[64, 32], final_activation=torch.tanh)
        self.optim = torch.optim.Adam(params=self.model.parameters(), lr=self.lr)

    def act(self, obs):
        with torch.no_grad():
            action = self.model.forward(obs).reshape(-1)
        return action.argmax().int().item()

    def learn(self, done):
        running_loss = 0
        n_expert_sample = self.expert_states.shape[0]
        final_batch = 1 if n_expert_sample % self.batch_size != 0 else 0
        for i in range(n_expert_sample // self.batch_size + final_batch):
            start_idx = i * self.batch_size
            end_idx = min(n_expert_sample, (i + 1) * self.batch_size)

            s = self.expert_states[start_idx:end_idx, :]
            a = self.expert_actions[start_idx:end_idx, :]

            output = self.model(s)
            loss = self.loss(output, a)
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()
            running_loss += loss.item()

        return {'loss': running_loss / (n_expert_sample // self.batch_size + final_batch)}

    def time_to_learn(self):
        if self.test:
            return False
        else:
            return True

    def load_expert_transition(self, file):
        with open(file, 'rb') as handle:
            expert_data = pickle.load(handle)
            expert_state = expert_data[:, :self.featureExtractor.outSize].contiguous()
            expert_actions = expert_data[:, self.featureExtractor.outSize:].contiguous()
        return expert_state, expert_actions
