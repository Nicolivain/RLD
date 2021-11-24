import torch

from Agents.Policy.PPO import AdaptativePPO


class ClippedPPO(AdaptativePPO):
    def __init__(self, env, opt, layers, k, epsilon=0.001, delta=1.5, loss='smoothL1', memory_size=1000, batch_size=1000, use_dkl=True, reversed_dkl=False):
        super(ClippedPPO, self).__init__(env, opt, layers, k ,delta, loss, memory_size, batch_size)
        self.epsilon = epsilon

    def _compute_objective(self, advantage, pi, new_pi, new_action_pi, action_pi):

        # on utilise torch.clamp pour le clipping
        clipped = torch.minimum(advantage * new_action_pi / action_pi,
                                advantage * torch.clamp(new_action_pi/action_pi, 1-self.epsilon, 1+self.epsilon))
        advantage_loss = -torch.mean(clipped)

        return advantage_loss

