from Agents.DQN.DQN import DQN
import torch


class ReplayDQN(DQN):
    """
    DQN with memory but no target network
    """
    def __init__(self, env, config, layers, loss='smoothL1', memory_size=10000, batch_size=100, **kwargs):
        super().__init__(env, config, layers, loss, memory_size=memory_size)
        self.batch_size  = batch_size

    def time_to_learn(self):
        if self.memory.nentities < self.memory_size:
            return False
        else:
            return super().time_to_learn()

    def learn(self, episode_done):
        batches     = self.memory.sample_batch(batch_size=self.batch_size)
        b_obs       = batches['obs']
        b_action    = batches['action']
        b_reward    = batches['reward'].float()
        b_next      = batches['new_obs']
        b_done      = batches['done']

        qhat = self.Q(b_obs)
        qvalues = torch.gather(qhat, -1, b_action.reshape(-1, 1))
        with torch.no_grad():
            next_qhat = self.Q(b_next)
            r = b_reward + self.discount * torch.max(next_qhat, dim=-1).values * (1 - b_done.float())

        loss = self.loss(qvalues, r)
        loss.backward()

        self.optim.step()
        self.optim.zero_grad()

        if episode_done:
            self.explo *= self.decay

        return {'Loss': loss.item()}
