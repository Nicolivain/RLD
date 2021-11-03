from Agents.DQN.DQN import DQN
import torch


class ReplayDQN(DQN):
    """
    DQN with memory but no target network
    """
    def __init__(self, env, config, layers, loss='smoothL1', memory_size=10000, batch_size=100):
        super().__init__(env, config, layers, loss, memory_size=memory_size)
        self.batch_size  = batch_size

    def time_to_learn(self):
        if self.memory.nentities < self.memory_size:
            return False
        else:
            return super().time_to_learn()

    def learn(self, episode_done):
        batches = self.memory.sample_batch(n_batch=self.memory_size // self.batch_size, batch_size=self.batch_size)
        b_obs = batches['obs']
        b_action = batches['action']
        b_reward = batches['reward']
        b_next = batches['new_obs']
        b_done = batches['done']

        train_loss = 0
        for i in range(self.memory_size // self.batch_size):
            obs, action, reward, new_obs, done = b_obs[i], b_action[i], b_reward[i].float(), b_next[i], b_done[i]

            qhat = self.Q(obs)
            next_qhat = self.Q(new_obs)

            r = reward + self.discount * torch.max(next_qhat, dim=-1).values * (1 - done.float())
            loss = self.loss(r, torch.gather(qhat, -1, action.reshape(-1, 1).long()))
            loss.backward()
            train_loss += loss.item()

            self.optim.step()
            self.optim.zero_grad()

        if episode_done:
            self.explo *= self.decay

        return train_loss / (self.memory_size // self.batch_size)
