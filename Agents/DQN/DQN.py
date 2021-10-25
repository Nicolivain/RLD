from Tools.core import NN
from Agents.Agent import Agent
from Tools.exploration import pick_greedy, pick_epsilon_greedy, pick_ucb
from Structure.Memory import Memory
import torch


class DQN(Agent):

    def __init__(self, env, config, layers, loss='smoothL1', memory_size=1, batch_size=100):
        super().__init__(env, config)

        self.featureExtractor = config.featExtractor(env)
        self.Q                = NN(self.featureExtractor.outSize, self.action_space.n, layers=layers, finalActivation=torch.nn.Tanh())
        self.loss             = torch.nn.SmoothL1Loss() if loss == 'smoothL1' else torch.nn.MSELoss()
        self.optim            = torch.optim.Adam(self.Q.parameters(), lr=self.alpha)
        self.memory           = Memory(memory_size)
        self.batch_size       = batch_size
        self.train_events     = 0

    def act(self, obs):
        values = self.Q(obs)
        if self.test:
            return pick_greedy(values)
        else:
            if self.exploMode == 0:
                return pick_epsilon_greedy(values, self.explo)
            elif self.exploMode == 1:
                return pick_ucb(values)
            else:
                raise NotImplementedError(f'{self.exploMode} does not correspond to any available exploration function')

    def learn(self, done):
        last_transition = self.memory.sample(1)
        obs     = last_transition['obs']
        action  = last_transition['action']
        reward  = last_transition['reward']
        done    = last_transition['done']

        qhat = self.Q(obs)
        r = reward + self.discount * torch.max(qhat) if not done else torch.Tensor([reward])
        loss = self.loss(r, qhat[0, action])
        loss.backward()

        self.train_events += 1
        if self.train_events == self.batch_size:
            self.optim.step()
            self.optim.zero_grad()
            self.train_events = 0

        if done:
            self.explo *= self.decay

    def store(self, transition):
        self.memory.store(transition)
