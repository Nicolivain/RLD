import numpy as np
import torch
from Agents.Agent import Agent
from torch import nn
from torch.nn import functional
from Structure.Memory import Memory


class QNet(nn.Module):
    """Q-values network used in the following VanillaDQN Agent"""
    def __init__(self, dim_in, dim_out, hidden_sizes):
        super().__init__()
        sizes = [dim_in] + hidden_sizes
        layers = []

        for i in range(len(sizes) - 1):
            layers += [nn.Linear(sizes[i], sizes[i + 1]),
                       nn.Tanh()]
        layers += [nn.Linear(sizes[-1], dim_out)]  # last layer

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class DQNGoal(Agent):

    def __init__(self, env, opt, layers, memory_size=10000, learning_rate=0.001, explo=0.1, explo_mode=0, discount=0.99, decay=0.9999, batch_size=1000, startEvents=2000, freq_update_target=1000, **kwargs):
        super().__init__(env, opt)

        self.featureExtractor = opt.featExtractor(env)
        self.memory           = Memory(memory_size, prior=False)
        self.memory_size      = memory_size
        self.batch_size = batch_size
        self.startEvents = startEvents

        self.learning_rate    = learning_rate
        self.explo            = explo
        self.explo_mode       = explo_mode
        self.decay            = decay
        self.discount         = discount

        self.freq_optim       = self.config.freqOptim
        self.n_events         = 0

        self.Q = QNet(2*self.featureExtractor.outSize, self.action_space.n, hidden_sizes=layers)
        self.target_net = QNet(2*self.featureExtractor.outSize, self.action_space.n, hidden_sizes=layers)
        self.target_net.load_state_dict(self.Q.state_dict())
        self.freq_update_target = freq_update_target

        self.optim = torch.optim.Adam(self.Q.parameters(), lr=learning_rate)

    def time_to_learn(self):
        if self.test:
            return False
        else:
            self.n_events +=1
            if self.n_events % self.freq_optim == 0 and self.memory.nentities > self.startEvents:
                return True

    def act(self, obs, goal):
        if np.random.uniform() < self.explo and (not self.test):
            return self.action_space.sample()
        else:
            with torch.no_grad():
                obs = torch.tensor(obs, dtype=torch.float32)
                goal = torch.tensor(goal, dtype=torch.float32)
                obs_goal = torch.cat([obs, goal], dim=0).view(1, -1)
                a = torch.argmax(self.Q(obs_goal)).item()
            return a

    def learn(self, done):

        # compute loss
        transitions = self.memory.sample_batch(self.batch_size)

        obs = transitions['obs']
        act = transitions['action']
        reward = transitions['reward']
        new_obs = transitions['new_obs']
        done = transitions['done']
        goal = transitions['goal']

        obs_goal = torch.cat([obs, goal], dim=1)  # (N, 4)
        next_obs_goal = torch.cat([new_obs, goal], dim=1)  # (N, 4)

        with torch.no_grad():
            q_next, _ = self.target_net(next_obs_goal).max(dim=1)  # (N,)
            q_target = reward + self.discount * q_next.unsqueeze(1) * (1 - done.float())  # (N, 1)

        q_pred = self.Q(obs_goal).gather(1, act)

        loss = nn.functional.smooth_l1_loss(q_pred, q_target.detach())

        # loss optim step
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        # decay after each episode
        self.explo *= self.decay

        return loss.item()

    def update_target_network(self):
        self.target_net.load_state_dict(self.Q.state_dict())

    def store(self, transition):
        self.memory.store(transition)
