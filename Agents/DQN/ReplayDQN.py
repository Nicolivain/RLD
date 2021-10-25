from Agents.DQN.DQN import DQN
import torch


class ReplayDQN(DQN):
    """
    DQN with memory but no target network
    """
    def __init__(self, env, config, layers, loss='smoothL1', memory_size=10000, batch_size=100):
        super().__init__(env, config, layers, loss, memory_size=memory_size)
        self.memory_size = memory_size
        self.batch_size  = batch_size

    def learn(self, done):
        batches = self.memory.sample_batch(n_batch=self.memory_size // self.batch_size, batch_size=self.batch_size)
        b_obs = batches['obs']
        b_action = batches['action']
        b_reward = batches['reward']
        b_done = batches['done']

        for i in range(self.memory_size // self.batch_size):
            obs, action, reward, done = b_obs[i], b_action[i], b_reward[i].float(), b_done[i]
            qhat = self.Q(obs)
            
            r = reward + self.discount * torch.max(qhat, dim=-1).values * (1 - done.float())

            loss = self.loss(r, torch.gather(qhat, -1, action.reshape(-1, 1).long()))
            loss.backward()

            self.optim.step()
            self.optim.zero_grad()

        if done:
            self.explo *= self.decay

