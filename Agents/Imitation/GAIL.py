import torch
import pickle
from Tools.core import NN
from Agents.Policy.PPO import AdaptativePPO
from Structure.Memory import Memory
from Tools.exploration import *


class GAIL(AdaptativePPO):
    def __init__(self, env, opt, expert_data_path, epsilon=0.01, learning_rate=0.001, discount=0.99, batch_size=100, disc_batch_size=50, memory_size=5000, **kwargs):
        super(GAIL, self).__init__(env, opt, learning_rate=learning_rate, discount=discount, batch_size=batch_size, memory_size=memory_size, **kwargs)
        self.epsilon = epsilon

        self.loss = torch.nn.SmoothL1Loss()
        self.disc_batch_size = disc_batch_size

        self.expert_states, self.expert_actions = self.__load_expert_transition(expert_data_path)

        self.discriminator = NN(self.featureExtractor.outSize + self.action_space.n, 1, layers=[64, 32], final_activation=torch.sigmoid)
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

        for i in range(n_disc_batch):
            start_idx = i * self.disc_batch_size
            end_idx = min(n_expert_sample, (i + 1) * self.disc_batch_size)

            s_expert = self.expert_states[start_idx:end_idx, :]
            a_expert = self.expert_actions[start_idx:end_idx, :]
            ipt = torch.cat((s_expert, a_expert), dim=1)
            expert_output = self.discriminator(ipt)

            real_batch = self.memory.sample_batch(batch_size=end_idx-start_idx)
            s_real = real_batch['obs']
            a_real = real_batch['action']
            ipt = torch.cat((s_real, self.__to_one_hot(a_real)), dim=1)
            sample_output = self.discriminator(ipt)

            loss = -(torch.log(expert_output) + torch.log(1 - sample_output)).mean()

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

        x = torch.cat([b_obs, self.__to_one_hot(b_action)], dim=1)
        adv_reward = self.discriminator(x)

        with torch.no_grad():
            # compute policy and value
            pi = self.model.policy(b_obs)
            action_pi = pi.gather(-1, b_action.reshape(-1, 1).long())

        avg_policy_loss = 0
        for i in range(self.k):
            # get the new action probabilities
            new_pi = self.model.policy(b_obs)
            new_action_pi = new_pi.gather(-1, b_action.reshape(-1, 1).long())

            # compute the objective with the new probabilities
            objective = self._compute_objective(adv_reward, pi, new_pi, new_action_pi, action_pi)
            avg_policy_loss += objective.item()

            # optimize
            self.optim.zero_grad()
            objective.backward()
            self.optim.step()

        # Updating betas
        self._update_betas(b_obs, pi)

        # updating value network with adversarial rewards
        loss = self._update_value_network(b_obs, adv_reward, b_new, b_done)

        #  reset memory
        del self.memory
        self.memory = Memory(mem_size=self.memory_size)

        epoch_dict['Average Policy Loss'] = avg_policy_loss / self.k
        epoch_dict['Value Loss'] = loss
        return epoch_dict

    def _compute_objective(self, advantage, pi, new_pi, new_action_pi, action_pi):

        # on utilise torch.clamp pour le clipping
        clipped = torch.minimum(advantage * new_action_pi / action_pi,
                                advantage * torch.clamp(new_action_pi / action_pi, 1 - self.epsilon, 1 + self.epsilon))
        advantage_loss = -torch.mean(clipped)

        return advantage_loss

    def _update_value_network(self, obs, adv_reward, next_obs, done):
        vs = self.model.value(obs)
        with torch.no_grad():
            target = adv_reward + vs.detach()
        loss = self.loss(vs, target)
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        return loss.item()

    def learn(self, done):
        # First we learn the disccriminator
        epoch_dict = self._train_discriminator()
        epoch_dict = self._ppo_step(epoch_dict)
        return epoch_dict

    def store(self, transition):
        if not self.test:
            self.memory.store(transition)

    def __load_expert_transition(self, file):
        with open(file, 'rb') as handle:
            expert_data = pickle.load(handle)
            expert_state = expert_data[:, :self.featureExtractor.outSize].contiguous()
            expert_actions = expert_data[:, self.featureExtractor.outSize:].contiguous()
        return expert_state, expert_actions

    def __to_one_hot(self, a):
        oha = torch.zeros(a.shape[0], self.action_space.n)
        oha[range(a.shape[0]), a.view(-1)] = 1
        return oha
