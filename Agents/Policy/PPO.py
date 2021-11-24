import torch

from Agents.Policy.A2C import A2C
from Structure.Memory import Memory
from Tools.distributions import batched_dkl


class AdaptativePPO(A2C):
    def __init__(self, env, opt, layers, k, delta=1e-3, loss='smoothL1', memory_size=1000, batch_size=1000):
        super(AdaptativePPO, self).__init__(env, opt, layers, loss, batch_size, memory_size)
        self.beta = 1
        self.delta = delta
        self.k = k
        self.min_beta = 1e-5

    def _update_betas(self, obs, old_pi):
        new_pi = self.model.policy(obs)
        dkl = batched_dkl(new_pi, old_pi)

        if dkl >= 1.5 * self.delta:
            self.beta *= 2
        elif dkl <= self.delta / 1.5:
            self.beta /= 2

        # clipping the value in case we go to low
        if self.beta < self.min_beta:
            self.beta = self.min_beta

    def _update_value_network(self, obs, reward, next_obs, done):
        with torch.no_grad():
            td0 = (reward + self.discount * self.model.critic(next_obs).squeeze() * (~done)).float()
        loss = self.loss(td0, self.model.critic(obs).squeeze())
        loss.backward()
        self.optim.step()
        self.optim.zero_grad()
        return loss.item()

    def _compute_objective(self, advantage, pi, new_pi, new_action_pi, action_pi):
        # compute l_theta_theta_k
        advantage_loss = torch.mean(advantage * new_action_pi / action_pi)

        # compute DKL_theta/theta_k
        dkl = batched_dkl(new_pi, pi)

        # computing the adjusted loss
        return -(advantage_loss - self.beta * dkl)

    def learn(self, done):
        batches = self.memory.sample_batch(batch_size=self.batch_size)

        b_obs = batches['obs']
        b_action = batches['action']
        b_reward = batches['reward']
        b_new = batches['new_obs']
        b_done = batches['done']

        with torch.no_grad():
            # compute policy and critic
            pi = self.model.policy(b_obs)
            values = self.model.critic(b_obs).squeeze()

            # compute td0
            next_critic = self.model.critic(b_new)
            td0 = (b_reward + self.discount * next_critic.squeeze() * (~b_done)).float()

            # compute advantage and action_probabilities
            advantage = td0 - values
            action_pi = pi.gather(-1, b_action.reshape(-1, 1).long()).squeeze()

        avg_policy_loss = 0
        for i in range(self.k):
            # get the new action probabilities
            new_pi = self.model.policy(b_obs)
            new_action_pi = new_pi.gather(-1, b_action.reshape(-1, 1).long()).squeeze()

            # compute the objective with the new probabilities
            objective = self._compute_objective(advantage, pi, new_pi, new_action_pi, action_pi)
            avg_policy_loss += objective.item()

            # optimize
            objective.backward()
            self.optim.step()
            self.optim.zero_grad()

        # Updating betas
        self._update_betas(b_obs, pi)

        # updating value network just like in A2C
        loss = self._update_value_network(b_obs, b_reward, b_new, b_done)

        #  reset memory
        del self.memory
        self.memory = Memory(mem_size=self.memory_size)

        return avg_policy_loss/self.k, loss
