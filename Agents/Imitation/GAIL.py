import torch
import pickle
from Tools.core import NN
from Agents.Agent import Agent
from Agents.Policy.PPO import PPONetwork
from Structure.Memory import Memory
from Tools.exploration import *


class GAIL(Agent):
    def __init__(self, env, config, expert_data, learning_rate=0.001, discount=0.99, batch_size=100, disc_batch_size=50, memory_size=5000, **kwargs):
        super(GAIL, self).__init__(env, config)

        self.featureExtractor = config.featExtractor(env)
        self.loss = torch.nn.SmoothL1Loss()
        self.bce  = torch.nn.BCEWithLogitsLoss()
        self.lr = learning_rate
        self.discount = discount
        self.batch_size = batch_size
        self.disc_batch_size = disc_batch_size

        self.expert_states  = expert_data[:, :self.featureExtractor.outSize]
        self.expert_actions = expert_data[:, self.featureExtractor.outSize:]

        self.discriminator = NN(self.featureExtractor.outSize, 2, layers=[64, 32], final_activation=torch.sigmoid)
        self.optim_d = self.optim = torch.optim.Adam(params=self.discriminator.parameters(), lr=self.lr)

        self.model = PPONetwork(self.featureExtractor.outSize, self.action_space.n, layers=[64, 32])
        self.optim = torch.optim.Adam(params=self.model.parameters(), lr=self.lr)

        self.memory = Memory(mem_size=memory_size)
        self.batch_size = batch_size if batch_size else memory_size
        self.memory_size = memory_size

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

            output = self.model(s_expert)
            loss = self.bce(output, labels)
            self.optim_d.zero_grad()
            loss.backward()
            self.optim_d.step()
            running_fake_disc_loss += loss.item()

        labels = torch.ones(self.disc_batch_size, dtype=torch.long)
        for i in range(n_disc_batch):
            s_real = self.memory.sample_batch(batch_size=self.disc_batch_size)['obs']
            output = self.model(s_real)
            loss = self.bce(output, labels)
            self.optim_d.zero_grad()
            loss.backward()
            self.optim_d.step()
            running_real_disc_loss += loss.item()

        epoch_dict = {'Disc fake loss': running_fake_disc_loss / n_disc_batch, 'Disc real loss': running_real_disc_loss / n_disc_batch, 'Total Disc loss': (running_real_disc_loss + running_fake_disc_loss) / n_disc_batch}
        return epoch_dict

    def learn(self, done):
        # First we learn the disccriminator
        epoch_dict = self._train_discriminator()
        # TODO: check that the discriminator trains properly
        # TODO: train the PPO with the new loss and stuff

        return epoch_dict

    def time_to_learn(self):
        if self.test:
            return False
        else:
            return True

    def store(self, transition):
        if not self.test:
            self.memory.store(transition)

    def load_expert_transition(self, file):
        with open(file, 'rb') as handle:
            expert_data = pickle.load(handle)
            expert_state = expert_data[:, :self.featureExtractor.outSize].contiguous()
            expert_actions = expert_data[:, self.featureExtractor.outSize:].contiguous()
        return expert_state, expert_actions
