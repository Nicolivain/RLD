import torch

from Agents.Policy.PPO import AdaptativePPO


class ClippedPPO(AdaptativePPO):
    def __init__(self, env, opt, layers, k, epsilon=0.01, memory_size=1000, batch_size=None, learning_rate=0.001, discount=0.99, **kwargs):
        super(ClippedPPO, self).__init__(env, opt, layers=layers, k=k, memory_size=memory_size, batch_size=batch_size, learning_rate=learning_rate, discount=discount, **kwargs)
        self.epsilon = epsilon

    def _compute_objective(self, advantage, pi, new_pi, new_action_pi, action_pi):

        # on utilise torch.clamp pour le clipping
        clipped = torch.minimum(advantage * new_action_pi / action_pi,
                                advantage * torch.clamp(new_action_pi/action_pi, 1-self.epsilon, 1+self.epsilon))
        advantage_loss = -torch.mean(clipped)

        return advantage_loss
