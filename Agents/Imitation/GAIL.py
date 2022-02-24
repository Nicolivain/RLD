import torch
import pickle
from Tools.core import NN
from Agents.Agent import Agent
from Agents.Policy.PPO import AdaptativePPO
from Structure.GailMemory import GailMemory
from Tools.exploration import *


class newGAIL(Agent):
    def __init__(self, env, opt, expert_data_path, epsilon=0.01, learning_rate=0.001, discount=0.99, batch_size=64, train_iter=1, k=1, entropy_weight=0.001, **kwargs):
        super().__init__(env, opt)

        self.expert_states, self.expert_actions = self.__load_expert_transition(expert_data_path)

        self.lr              = learning_rate
        self.epsilon         = epsilon
        self.discount        = discount
        self.batch_size      = batch_size
        self.train_iter      = train_iter  # number of sampled batch per training
        self.k               = k           # number of times a batch is passed in ppo
        self.entropy_weight  = entropy_weight

        self.bce = torch.nn.BCELoss()
        self.smoothL1 = torch.nn.SmoothL1Loss()

        self.discriminator = NN(self.featureExtractor.outSize + self.action_space.n, 1, layers=[64, 32], activation=torch.tanh, final_activation=torch.nn.Softmax(dim=-1))
        self.optim_d = torch.optim.Adam(params=self.discriminator.parameters(), lr=self.lr)

        self.policy = NN(self.featureExtractor.outSize, self.action_space.n, layers=[64, 32], activation=torch.tanh, final_activation=torch.nn.Softmax(dim=-1))
        self.optim_pi = torch.optim.Adam(params=self.policy.parameters(), lr=self.lr)

        self.value = NN(self.featureExtractor.outSize, 1, layers=[64, 32], activation=torch.tanh, final_activation=None)
        self.optim_v = torch.optim.Adam(params=self.value.parameters(), lr=self.lr)

        self.memory = GailMemory(self.batch_size)

    def act(self, obs):
        p = self.policy(obs)
        multinomial = Categorical(logits=p)
        return multinomial.sample().item()

    def learn(self, done):
        running_ld = 0
        running_lp = 0
        running_lv = 0
        running_entropy = 0
        for i in range(self.train_iter):
            self.memory.compute_cumulated_r(self, self.action_space.n)  # compute the cumulated rewards with the disc
            agent_batch = self.memory.get_minibatch_proxy_reward()

            agent_obs = agent_batch['obs']
            agent_act = agent_batch['action']
            agent_cum_rewards = agent_batch['reward']

            indices = np.random.choice(range(len(self.expert_states)), self.batch_size)
            expert_obs = self.expert_states[indices, :]
            expert_act = self.expert_actions[indices, :]

            loss_d = self._discriminator_step(expert_obs, expert_act, agent_obs, agent_act)
            loss_p, loss_v, entropy = self._ppo_step(agent_obs, agent_act.long(), agent_cum_rewards)

            running_ld += loss_d
            running_lp += loss_p
            running_lv += loss_v
            running_entropy += entropy

        self.memory.reset()
        return {'Disc loss': running_ld/self.train_iter, 'Policy loss': running_lp/self.train_iter, 'Value loss': running_lv/self.train_iter, 'Entropy': running_entropy/self.train_iter}

    def _discriminator_step(self, expert_obs, expert_act, agent_obs, agent_act):
        # first encode the action data
        one_hot_agent_act = self.__to_one_hot(agent_act)

        input_expert = torch.cat([expert_obs, expert_act], dim=1)
        noise = torch.normal(0, 0.01, size=input_expert.shape)
        d_out = self.discriminator(input_expert + noise)
        loss = self.bce(d_out, torch.ones_like(d_out))

        input_agent = torch.cat([agent_obs, one_hot_agent_act], dim = 1)
        noise = torch.normal(0, 0.01, size=input_agent.shape)
        d_out = self.discriminator(input_agent + noise)
        loss = loss + self.bce(d_out, torch.zeros_like(d_out))

        assert not loss.isnan()
        self.optim_d.zero_grad()
        loss.backward()
        self.optim_d.step()
        return loss.item()

    def _ppo_step(self, obs, act, disc_cumulative_rewards):
        with torch.no_grad():
            sampling_policy = self.policy(obs)
            adversarial_advantage = disc_cumulative_rewards - self.value(obs)
        sampling_action_prob = sampling_policy.gather(1, act)
        for s in range(self.k):
            pi = self.policy(obs)
            action_prob = pi.gather(1, act)
            ratio = action_prob / sampling_action_prob

            entropy = Categorical(pi).entropy()
            loss_pi = (-torch.clamp(ratio,  min=1-self.epsilon, max=1+self.epsilon) * adversarial_advantage.view(-1) - self.entropy_weight * entropy).mean()
            self.optim_pi.zero_grad()
            loss_pi.backward()
            self.optim_pi.step()

            # train the critic
            loss_v = self.smoothL1(self.value(obs), disc_cumulative_rewards)
            self.optim_v.zero_grad()
            loss_v.backward()
            self.optim_v.step()

            return loss_pi.item(), loss_v.item(), entropy.mean().item()

    def store(self, transition):
        if not self.test:
            self.memory.store(transition)

    def time_to_learn(self):
        return (not self.test) and self.batch_size <= len(self.memory)

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
