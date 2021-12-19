from Agents.DQN.DQN import DQN
import torch
from copy import deepcopy


class TargetDQN(DQN):
    """
    DQN with target network but no memory (ie: Vanilla DQN with target network)
    """
    def __init__(self, env, opt, layers, learning_rate=0.001, explo=0.01, explo_mode=0, discount=0.99, freq_update_target=1000, decay=0.9999, **kwargs):
        super().__init__(env, opt, layers, memory_size=1,  learning_rate=learning_rate, explo=explo, explo_mode=explo_mode, discount=discount, decay=decay)
        self.target_net = deepcopy(self.Q)
        self.update_target = freq_update_target
        self.n_learn = 0

    def learn(self, episode_done):
        last_transition = self.memory.sample_batch(1)
        obs     = last_transition['obs']
        action  = last_transition['action']
        reward  = last_transition['reward']
        new_obs = last_transition['new_obs']
        done    = last_transition['done']

        qhat = self.Q(obs)
        with torch.no_grad():
            qhat_target = self.target_net(new_obs)
            r = reward + self.discount * torch.max(qhat_target) if not done else torch.Tensor([reward])

        loss = self.loss(r, qhat[0, action.long().item()])
        loss.backward()

        self.optim.step()
        self.optim.zero_grad()

        self.n_learn += 1
        if self.n_learn % self.update_target == 0:
            print('Target net updated')
            self.target_net.load_state_dict(self.Q.state_dict())

        if episode_done:
            self.explo *= self.decay

        return {'Loss': loss.item()}
