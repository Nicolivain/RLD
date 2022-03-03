from Agents.DQN.TargetDQN import TargetDQN
import torch


class DoubleDQN(TargetDQN):
    """Double Q-learning uses Q-network to select actions and Q-network target to evaluate the selected actions"""

    def __init__(self, env, opt, layers, learning_rate=0.001, explo=0.01, explo_mode=0, discount=0.99, freq_update_target=1000, decay=0.9999, memory_size=1, **kwargs):
        super().__init__(env, opt, layers, memory_size=memory_size,  learning_rate=learning_rate, explo=explo, explo_mode=explo_mode, discount=discount, decay=decay)

    def learn(self, episode_done):
        last_transition = self.memory.sample_batch(1)

        bs = last_transition['obs'].shape[0]
        obs     = last_transition['obs'].view(bs, -1)
        action  = last_transition['action'].view(bs, -1)
        reward  = last_transition['reward'].view(bs, -1)
        new_obs = last_transition['new_obs'].view(bs, -1)
        done    = last_transition['done'].view(bs, -1)

        q_values = self.Q(obs).gather(dim=1, index=action)  # online q_values
        with torch.no_grad():
            # no greedy action using the target but the online network
            _, act = self.Q(new_obs).max(dim=1)
            act = act.unsqueeze(1)
            res_target = self.target_net(new_obs).gather(dim=1, index=act)
            target_q_values = reward + self.discount * res_target * (~done).float()

        loss = self.loss(target_q_values, q_values)
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