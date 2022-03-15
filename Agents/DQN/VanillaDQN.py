import torch

from Agents.Agent import Agent
from Structure.Memory import Memory
from Tools.core import NN
from Tools.exploration import pick_greedy, pick_epsilon_greedy, pick_ucb


class VanillaDQN(Agent):

    def __init__(self, env, opt, layers, memory_size=10000, learning_rate=0.001, explo=0.1, explo_mode=0, discount=0.99, decay=0.9999, **kwargs):
        super().__init__(env, opt)

        self.featureExtractor = opt.featExtractor(env)
        self.Q                = NN(self.featureExtractor.outSize, self.action_space.n, layers=layers, final_activation=torch.nn.ReLU(), activation=torch.nn.ReLU())
        self.loss             = torch.nn.SmoothL1Loss()
        self.optim            = torch.optim.Adam(self.Q.parameters(), lr=learning_rate)
        self.memory           = Memory(memory_size)
        self.memory_size      = memory_size

        self.learning_rate    = learning_rate
        self.explo            = explo
        self.explo_mode       = explo_mode
        self.decay            = decay
        self.discount         = discount

        self.freq_optim       = self.config.freqOptim
        self.n_events         = 0

    def time_to_learn(self):
        if self.test:
            return False
        else:
            self.n_events += 1
            if self.n_events % self.freq_optim == 0:
                return True

    def act(self, obs):
        with torch.no_grad():
            values = self.Q(obs).numpy().reshape(-1)
        if self.test:
            return pick_greedy(values)
        else:
            if self.explo_mode == 0:
                return pick_epsilon_greedy(values, self.explo)
            elif self.explo_mode == 1:
                return pick_ucb(values)
            else:
                raise NotImplementedError(f'{self.exploMode} does not correspond to any available exploration function')

    def learn(self, done):
        last_transition = self.memory.sample_batch(1)
        obs     = last_transition['obs']
        action  = last_transition['action']
        reward  = last_transition['reward']
        new_obs = last_transition['new_obs']
        done    = last_transition['done']

        qhat = self.Q(obs)

        with torch.no_grad():
            next_qhat = self.Q(new_obs)
            r = reward + self.discount * torch.max(next_qhat) if not done else torch.Tensor([reward])

        loss = self.loss(qhat[0, action.long().item()], r)
        loss.backward()

        self.optim.step()
        self.optim.zero_grad()

        if done:
            self.explo *= self.decay

        return {'Loss': loss.item()}

    def store(self, transition):
        if not self.test:
            self.memory.store(transition)
