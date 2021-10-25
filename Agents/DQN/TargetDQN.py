from Agents.DQN.DQN import DQN
from Tools.core import NN
import torch


class TargetDQN(DQN):
    """
    DQN with target network but no memory (ie: Vanilla DQN with target network)
    """
    def __init__(self, env, config, layers, loss='smoothL1'):
        super().__init__(env, config, layers, loss, memory_size=1)
        self.target_net = NN(self.featureExtractor.outSize, self.action_space.n, layers=layers)

    def learn(self, done):
        last_transition = self.memory.sample(1)[-1][0]
        obs     = last_transition['obs']
        action  = last_transition['action']
        reward  = last_transition['reward']
        done    = last_transition['done']

        qhat = self.Q(obs)

        with torch.no_grad():
            qhat_target = self.target_net(obs)
            r = reward + self.discount * torch.max(qhat_target, dim=-1).values * (1 - done.float())

        loss = self.loss(r, torch.gather(qhat, -1, action.reshape(-1, 1).long()))
        loss.backward()

        self.optim.step()
        self.optim.zero_grad()

        if done:
            self.explo *= self.decay
