from Agents.DQN.VanillaDQN import VanillaDQN
import torch


class ReplayDQN(VanillaDQN):
    """
    VanillaDQN with memory but no target network
    """
    def __init__(self, env, opt, layers, memory_size=10000, batch_size=100, learning_rate=0.001, explo=0.01, explo_mode=0, discount=0.99, decay=0.9999, **kwargs):
        super().__init__(env, opt, layers, memory_size=memory_size, learning_rate=learning_rate, explo=explo, explo_mode=explo_mode, discount=discount, decay=decay)
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
        qvalues = torch.gather(qhat, -1, b_action.long().reshape(-1, 1))
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
