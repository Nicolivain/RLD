from Agents.DQN.DQN import DQN
from Tools.core import NN
import torch
from copy import deepcopy


class MinDQN(DQN):
    def __init__(self, env, opt, layers, memory_size=10000, batch_size=100, freq_update_target=100, learning_rate=0.001, explo=0.1, explo_mode=0, discount=0, decay=0.9999, **kwargs):
        super().__init__(env, opt, layers, memory_size=memory_size,  learning_rate=learning_rate, explo=explo, explo_mode=explo_mode, discount=discount, decay=decay)
        self.batch_size = batch_size
        self.target_net = deepcopy(self.Q)
        self.update_target = freq_update_target
        self.n_learn = 0

    def time_to_learn(self):
        if self.memory.nentities < self.memory_size:
            return False
        else:
            return super().time_to_learn()

    def learn(self, episode_done):
        batches = self.memory.sample_batch(batch_size=self.batch_size)
        b_obs       = batches['obs']
        b_action    = batches['action']
        b_reward    = batches['reward'].float()
        b_next      = batches['new_obs']
        b_done      = batches['done']

        qhat = self.Q(b_obs)
        learning = torch.gather(qhat, -1, b_action.long().reshape(-1, 1)).squeeze()

        with torch.no_grad():
            qhat_target = self.target_net(b_next)
            r = b_reward + self.discount * torch.max(qhat_target, dim=-1).values * (1-b_done.float())

        loss = self.loss(learning, r)
        loss.backward()

        self.optim.step()
        self.optim.zero_grad()

        self.n_learn += 1
        if self.n_learn % self.update_target == 0:
            self.target_net.load_state_dict(self.Q.state_dict())

        if episode_done:
            self.explo *= self.decay

        return {'Loss': loss.item()}
