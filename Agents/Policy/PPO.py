import torch

from Agents.Policy.A2C import A2C
from Structure.Memory import Memory


class AdaptativePPO(A2C):
    def __init__(self, env, opt, layers, k, delta=1.5, loss='smoothL1', memory_size=1000, batch_size=1000):
        super(AdaptativePPO, self).__init__(env, opt, layers, loss, batch_size, memory_size)
        self.beta = 0
        self.delta = delta
        self.k = k

    def _update_betas(self, obs, actions, old_action_pi):
        new_pi = self.model.policy(obs)
        new_action_pi = new_pi.gather(-1, actions.reshape(-1, 1).long()).squeeze()

        dkl = torch.distributions.kl_divergence(
            torch.distributions.Categorical(old_action_pi),
            torch.distributions.Categorical(new_action_pi)
        ).mean()

        if dkl >= 1.5 * self.delta:
            self.beta *= 2
        elif dkl <= self.delta / 1.5:
            self.beta /= 2

    def _update_value_network(self, td, values):
        loss = self.loss(td, values)
        loss.backward()
        self.optim.step()
        self.optim.zero_grad()
        return loss.item()

    def _compute_objective(self, advantage, new_action_pi, action_pi):
        # compute l_theta_theta_k
        advantage_loss = torch.mean(advantage * new_action_pi / action_pi)

        # compute DKL_theta/theta_k
        dkl = torch.distributions.kl_divergence(
            torch.distributions.Categorical(action_pi),
            torch.distributions.Categorical(new_action_pi)
        ).mean()

        # computing the adjusted loss
        return advantage_loss - self.beta * dkl

    def learn(self, done):
        batches = self.memory.sample_batch(batch_size=self.batch_size)

        b_obs = batches['obs']
        b_action = batches['action']
        b_reward = batches['reward']
        b_new = batches['new_obs']
        b_done = batches['done']

        pi = self.model.policy(b_obs)
        critic = self.model.critic(b_obs).squeeze()
        with torch.no_grad():
            # compute td0
            next_critic = self.model.critic(b_new)
            td0 = (b_reward + self.discount * next_critic.squeeze() * (~b_done)).float()

            # compute advantage and action_probabilities
            advantage = td0 - critic
            action_pi = pi.gather(-1, b_action.reshape(-1, 1).long()).squeeze()

        for i in range(self.k):
            # get the new action probabilities
            new_pi = self.model.policy(b_obs)
            new_action_pi = new_pi.gather(-1, b_action.reshape(-1, 1).long()).squeeze()

            # compute the objective with the new probabilities
            objective = self._compute_objective(advantage, new_action_pi, action_pi)

            # optimize
            objective.backward()
            self.optim.step()
            self.optim.zero_grad()

        # Updating betas
        self._update_betas(b_obs, b_action, action_pi)

        # updating value network just like in A2C
        loss = self._update_value_network(td0, critic)

        #  reset memory
        del self.memory
        self.memory = Memory(mem_size=self.memory_size)

        return
