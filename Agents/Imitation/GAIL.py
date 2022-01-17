import torch
from Tools.core import NN
from Agents.Agent import Agent
from Tools.exploration import *


class GAIL(Agent):
    def __init__(self, env, config, expert_data, learning_rate=0.001, discount=0.99, batch_size=100, **kwargs):
        super(GAIL, self).__init__(env, config)

        self.featureExtractor = config.featExtractor(env)
        self.loss = torch.nn.SmoothL1Loss()
        self.lr = learning_rate
        self.discount = discount
        self.batch_size = batch_size

        self.expert_states  = expert_data[:, :self.featureExtractor.outSize]
        self.expert_actions = expert_data[:, self.featureExtractor.outSize:]

        self.model = NN(self.featureExtractor.outSize, self.action_space.n, layers=[64, 32], final_activation=torch.tanh)
        self.optim = torch.optim.Adam(params=self.model.parameters(), lr=self.lr)

    def act(self, obs):
        with torch.no_grad():
            action = self.model.policy(obs).reshape(-1)
        return action

    def learn(self, done):
        pass