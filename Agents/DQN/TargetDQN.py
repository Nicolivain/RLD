from Agents.DQN.DQN import DQN
import torch
from copy import deepcopy


class TargetDQN(DQN):
    """
    DQN with target network but no memory (ie: Vanilla DQN with target network)
    """
    def __init__(self, env, config, layers, loss='smoothL1', update_target=100):
        super().__init__(env, config, layers, loss, memory_size=1)
        self.target_net = deepcopy(self.Q)
        self.update_target = update_target
        self.n_learn = 0

    def learn(self, episode_done):
        last_transition = self.memory.sample(1)[-1][0]
        obs     = last_transition['obs']
        action  = last_transition['action']
        reward  = last_transition['reward']
        new_obs = last_transition['new_obs']
        done    = last_transition['done']

        qhat = self.Q(obs)

        with torch.no_grad():
            qhat_target = self.target_net(new_obs)
            r = reward + self.discount * torch.max(qhat_target) if not done else torch.Tensor([reward])

        loss = self.loss(r, qhat[0, action])
        loss.backward()

        self.optim.step()
        self.optim.zero_grad()

        self.n_learn += 1
        if self.n_learn % self.update_target == 0:
            print('Target net updated')
            self.target_net.load_state_dict(self.Q.state_dict())

        if episode_done:
            self.explo *= self.decay

        return loss.item()
