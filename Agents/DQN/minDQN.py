from Agents.DQN.DQN import DQN
from Tools.core import NN
import torch


class MinDQN(DQN):
    def __init__(self, env, config, layers, loss='smoothL1', memory_size=10000, batch_size=100):
        super().__init__(env, config, layers, loss, memory_size)
        self.batch_size = batch_size
        self.target_net = NN(self.featureExtractor.outSize, self.action_space.n, layers=layers)

    def time_to_learn(self):
        if self.memory.nentities < self.memory_size:
            return False
        else:
            return super().time_to_learn()

    def learn(self, done):
        batches = self.memory.sample_batch(n_batch=self.memory_size//self.batch_size, batch_size=self.batch_size)
        b_obs       = batches['obs']
        b_action    = batches['action']
        b_reward    = batches['reward']
        b_done      = batches['done']

        for i in range(self.memory_size//self.batch_size):
            obs, action, reward, done = b_obs[i], b_action[i], b_reward[i].float(), b_done[i]
            qhat = self.Q(obs)

            with torch.no_grad():
                qhat_target = self.target_net(obs)
                r = reward + self.discount * torch.max(qhat_target, dim=-1).values * (1-done.float())

            loss = self.loss(r, torch.gather(qhat, -1, action.reshape(-1, 1).long()))
            loss.backward()

            self.optim.step()
            self.optim.zero_grad()

        if done:
            self.explo *= self.decay
