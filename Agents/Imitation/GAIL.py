import torch
import pickle
from Tools.core import NN
from Agents.Agent import Agent
from Agents.Policy.PPO import AdaptativePPO
from Structure.Memory import Memory
from Tools.exploration import *
from Tools.distributions import batched_dkl


class GAIL(AdaptativePPO):
    def __init__(self, env, config, expert_data, learning_rate=0.001, discount=0.99, batch_size=100, disc_batch_size=50, memory_size=5000, **kwargs):
        super(GAIL, self).__init__(env, config, learning_rate=learning_rate, discount=discount, batch_size=batch_size, memory_size=memory_size, **kwargs)

        self.loss = torch.nn.SmoothL1Loss()
        self.bce  = torch.nn.BCEWithLogitsLoss()
        self.disc_batch_size = disc_batch_size

        self.expert_states  = expert_data[:, :self.featureExtractor.outSize]
        self.expert_actions = expert_data[:, self.featureExtractor.outSize:]

        self.discriminator = NN(self.featureExtractor.outSize + self.action_space.n, 2, layers=[64, 32], final_activation=torch.sigmoid)
        self.optim_d = self.optim = torch.optim.Adam(params=self.discriminator.parameters(), lr=self.lr)

    def act(self, obs):
        with torch.no_grad():
            action = self.model.policy(obs).reshape(-1)
        return pick_sample(action)

    def _train_discriminator(self):
        running_fake_disc_loss = 0
        running_real_disc_loss = 0
        n_expert_sample = self.expert_states.shape[0]
        final_batch = 1 if n_expert_sample % self.batch_size != 0 else 0
        n_disc_batch = n_expert_sample // self.disc_batch_size + final_batch
        labels = torch.zeros(self.disc_batch_size, dtype=torch.long)

        for i in range(n_disc_batch):
            start_idx = i * self.disc_batch_size
            end_idx = min(n_expert_sample, (i + 1) * self.disc_batch_size)

            s_expert = self.expert_states[start_idx:end_idx, :]
            a_expert = self.expert_actions[start_idx:end_idx, :]
            ipt = torch.cat((s_expert, a_expert), dim=1)

            output = self.model(ipt)
            loss = self.bce(output, labels)
            self.optim_d.zero_grad()
            loss.backward()
            self.optim_d.step()
            running_fake_disc_loss += loss.item()

        labels = torch.ones(self.disc_batch_size, dtype=torch.long)
        for i in range(n_disc_batch):
            real_batch = self.memory.sample_batch(batch_size=self.disc_batch_size)
            s_real = real_batch['obs']
            a_real = real_batch['action']
            ipt = torch.cat((s_real, a_real), dim=1)

            output = self.model(ipt)
            loss = self.bce(output, labels)
            self.optim_d.zero_grad()
            loss.backward()
            self.optim_d.step()
            running_real_disc_loss += loss.item()

        epoch_dict = {'Disc fake loss': running_fake_disc_loss / n_disc_batch, 'Disc real loss': running_real_disc_loss / n_disc_batch, 'Total Disc loss': (running_real_disc_loss + running_fake_disc_loss) / n_disc_batch}
        return epoch_dict

    def _ppo_step(self, epoch_dict):
        batches = self.memory.sample_batch(batch_size=self.batch_size)

        bs = batches['obs'].shape[0]
        b_obs = batches['obs'].view(bs, -1)
        b_action = batches['action'].view(bs, -1)
        b_reward = batches['reward'].view(bs, -1)
        b_new = batches['new_obs'].view(bs, -1)
        b_done = batches['done'].view(bs, -1)

        #TODO compute the adversarial cost
        #TODO check losses to make sur they match

        with torch.no_grad():
            # compute policy and value
            pi = self.model.policy(b_obs)
            values = self.model.value(b_obs)

            # compute td0
            next_value = self.model.value(b_new)
            td0 = (b_reward + self.discount * next_value * (~b_done)).float()

            # compute advantage and action_probabilities
            advantage = td0 - values
            action_pi = pi.gather(-1, b_action.reshape(-1, 1).long())

        avg_policy_loss = 0
        for i in range(self.k):
            # get the new action probabilities
            new_pi = self.model.policy(b_obs)
            new_action_pi = new_pi.gather(-1, b_action.reshape(-1, 1).long())

            # compute the objective with the new probabilities
            objective = self._compute_objective(advantage, pi, new_pi, new_action_pi, action_pi)
            avg_policy_loss += objective.item()

            # optimize
            self.optim.zero_grad()
            objective.backward()
            self.optim.step()

        # Updating betas
        self._update_betas(b_obs, pi)

        # updating value network with adversarial rewards
        # TODO
        loss = self._update_value_network(b_obs, b_reward, b_new, b_done)

        #  reset memory
        del self.memory
        self.memory = Memory(mem_size=self.memory_size)

        epoch_dict['Average Policy Loss'] = avg_policy_loss / self.k
        epoch_dict['Value Loss'] = loss
        epoch_dict['Beta'] = self.beta
        return epoch_dict

    def _compute_objective(self, advantage, pi, new_pi, new_action_pi, action_pi):
        # compute l_theta_theta_k
        advantage_loss = torch.mean(advantage * new_action_pi / action_pi)

        # compute DKL_theta/theta_k
        dkl = 0
        if self.dkl:
            if self.reversed:
                # if we want to experiment with the reversed dkl
                dkl = batched_dkl(pi, new_pi)
            else:
                dkl = batched_dkl(new_pi, pi)

        # computing the adjusted loss
        return -(advantage_loss - self.beta * dkl)

    def learn(self, done):
        # First we learn the disccriminator
        epoch_dict = self._train_discriminator()
        epoch_dict = self._ppo_step(epoch_dict)
        return epoch_dict

    def store(self, transition):
        if not self.test:
            self.memory.store(transition)

    def load_expert_transition(self, file):
        with open(file, 'rb') as handle:
            expert_data = pickle.load(handle)
            expert_state = expert_data[:, :self.featureExtractor.outSize].contiguous()
            expert_actions = expert_data[:, self.featureExtractor.outSize:].contiguous()
        return expert_state, expert_actions
