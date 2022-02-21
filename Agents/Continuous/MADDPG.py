from copy import deepcopy

import torch

from Agents.Agent import Agent
from Structure.Memory import Memory
from Tools.core import NN
from Tools.core import Orn_Uhlen
from Agents.Continuous.DDPG import *
from Tools.core import NothingToDo

class QPolicyNet_MA():
    def __init__(self, in_size, action_space_size, layers, final_activation_q=None, final_activation_p=torch.nn.Tanh(), activation_q=torch.nn.LeakyReLU(), activation_p=torch.nn.LeakyReLU(), dropout=0):

        self.q_net      = NN(in_size + action_space_size, 1, layers=layers, final_activation=final_activation_q, activation=activation_q, dropout=dropout, batchnorm = True)
        self.policy_net = NN(in_size, action_space_size, layers=layers, final_activation=final_activation_p, activation=activation_p, dropout=dropout, batchnorm = True)

    def policy(self, x):
        x = x.float()
        return self.policy_net(x)

    def q(self, obs, action):
        action = action.float()
        obs = obs.float()
        ipt = torch.cat([obs, action], dim=-1)
        return self.q_net(ipt)
#
class DDPG_adapt():
    def __init__(self, dim_obs, dim_act , layers=[10, 10], lr_q=0.001, lr_policy=0.0005, explo=0.1, **kwargs):

        self.network        = QPolicyNet_MA(in_size=dim_obs, action_space_size=dim_act, layers=layers)
        self.target_net     = deepcopy(self.network)
        self.optim_q        = torch.optim.Adam(params=self.network.q_net.parameters(), lr=lr_q)
        self.optim_policy   = torch.optim.Adam(params=self.network.policy_net.parameters(), lr=lr_policy)

        #La memory est maintenant dans le Multi-DDPG, on veut stocker les obs et actions coordonnées (pas une memory par DDPG, mais une memory globale)
        self.noise = Orn_Uhlen(n_actions=dim_act, sigma=explo)

        self.loss = torch.nn.SmoothL1Loss()

class MADDPG():
    """Dans le cas du MADDPG, il faut utiliser un Q qui prend en compte les paramètres des autres DDPG
    pour une optimisation "collaborative" de tous les DDPG
    """
    def __init__(self, env, world, opt, layers=[10, 10], batch_per_learn=10, batch_size=1000, memory_size=1000000, lr_q=[0.001], lr_policy=[0.0005], discount=0.99, rho=0.01, start_events=100000, explo=0.1, **kwargs):
        #super().__init__(env, opt) #seulement si on fait un héritage de agent pour obtenir le self.action_space et self.test

        self.n_agents = len(env.agents)
        # Compute the dimensions of the env
        junk = env.reset()
        self.obs_size = len(junk[0])
        self.action_size = world.dim_p

        self.agents = [DDPG_adapt(self.obs_size, self.action_size , layers=layers, lr_q=lr_q[k], lr_policy=lr_policy[k], explo=explo) for k in range(len(lr_q))]

        self.config = opt
        self.featureExtractor = NothingToDo(env) #opt.featExtractor(env)
        self.loss = torch.nn.SmoothL1Loss()
        self.p_lr = lr_policy #list
        self.q_lr = lr_q #list
        self.discount = discount
        self.explo = explo

        self.test = False
        self.memory = Memory(mem_size=memory_size)
        self.batch_size = batch_size
        self.memory_size = memory_size
        self.batch_per_learn = batch_per_learn
        self.startEvents = start_events
        self.noise = Orn_Uhlen(n_actions=self.action_size, sigma=self.explo)

        self.rho = rho

        self.freq_optim = self.config.freqOptim
        self.n_events   = 0

        self.min = -1.
        self.max = 1.

    def store(self, transition):
        if not self.test:
            self.memory.store(transition)

    def act(self, obs):
        if self.n_events < self.startEvents:
            actions = torch.zeros(self.action_size, self.n_agents)
        else:
            with torch.no_grad():
                actions = torch.tensor([agent.network.policy(o) for agent,o in zip(self.agents, obs)])
        n = torch.zeros(1)
        if not self.test:
            n = self.noise.sample()
            # n = torch.rand(1) * 0.1
        o = (torch.transpose(actions, 0, 1)+n).squeeze()
        return torch.clamp(o, self.min, self.max)

    def time_to_learn(self):
        self.n_events += 1
        if self.n_events % self.freq_optim != 0 or self.test:
            return False
        else:
            return True

    def learn(self, episode_done):
        mean_qloss, mean_ploss = 0, 0
        for k in range(self.batch_per_learn):
            qloss, ploss = self._train_batch()
            mean_qloss += qloss
            mean_ploss += ploss
        return {'Q Loss': mean_qloss/self.batch_per_learn, 'Policy Loss': mean_ploss/self.batch_per_learn}

    def _train_batch(self):
        batches = self.memory.sample_batch(batch_size=self.batch_size)

        bs = batches['obs'].shape[0]
        b_obs = [batches['obs'][:,k,:].view(bs, -1) for k in range(self.n_agents)]
        #print(batches['action'].shape, batches['reward'].shape,batches['new_obs'].shape, batches['done'].shape)
        b_action = torch.split(batches['action'],1000)
        #b_action = [batches['action'][:,k,:].view(bs, -1) for k in range(self.n_agents)]
        b_reward = [batches['reward'][:,:,k].view(bs, -1) for k in range(self.n_agents)]
        b_new = [batches['new_obs'][:,k,:].view(bs, -1) for k in range(self.n_agents)]
        b_done = [batches['done'][:,:,k].view(bs, -1) for k in range(self.n_agents)]

        # update q net
        q_loss = self._update_q_all(b_obs, b_action, b_reward, b_new, b_done)

        # update policy
        loss_policy = self._update_policy_all(b_obs)

        # update target network
        self._update_target_all()

        return q_loss.item(), loss_policy.item()

    def _update_q_all(self, b_obs, b_action, b_reward, b_new, b_done):
        q_loss_all = 0
        # on commence par calculer la target
        for i,agent in enumerate(self.agents) :
            with torch.no_grad():
                target_next_act = agent.target_net.policy(b_new[i])
                target = b_reward[i] + self.discount * agent.target_net.q(b_new[i], target_next_act) * (~b_done[i]).float()
            preds_q = agent.network.q(b_obs[i], b_action[i])
            q_loss = agent.loss(preds_q, target)
            q_loss_all += q_loss
            agent.optim_q.zero_grad()
            q_loss.backward()
            agent.optim_q.step()
        return q_loss_all

    def _update_policy_all(self, b_obs):
        loss_policy_all = 0
        for i,agent in enumerate(self.agents):
            pred_act = agent.network.policy(b_obs[i])
            pred_q = agent.network.q(b_obs[i], pred_act)
            loss_policy= -torch.mean(pred_q)
            loss_policy_all += loss_policy
            agent.optim_policy.zero_grad()
            loss_policy.backward()
            agent.optim_policy.step()

        return loss_policy_all

    def _update_target_all(self):
        """soft update, using rho"""
        with torch.no_grad():
            for agent in self.agents :
                for target_p, net_p in zip(agent.target_net.policy_net.parameters(), agent.network.policy_net.parameters()):
                    new_p = (1 - self.rho) * target_p + self.rho * net_p
                    target_p.copy_(new_p)

                for target_p, net_p in zip(agent.target_net.q_net.parameters(), agent.network.q_net.parameters()):
                    new_p = (1 - self.rho) * target_p + self.rho * net_p
                    target_p.copy_(new_p)

    def save(self, path):
        # TODO: fix the save bug
        """
        Unknown bug: can't pickle NoneType ??
        f = open(path, 'wb')
        pickle.dump(self, f)
        f.close()
        """
        pass
