from copy import deepcopy

import torch

from Agents.Agent import Agent
from Structure.Memory import Memory
from Tools.core import NN
from Tools.core import Orn_Uhlen
from Agents.Continuous.DDPG import *
from Tools.core import *

class QPolicyNet_MA():
    def __init__(self, n_agents, in_size, action_space_size, layers_p, layers_q, final_activation_q=None, final_activation_p=torch.nn.Tanh(), activation_q=torch.nn.LeakyReLU(), activation_p=torch.nn.LeakyReLU(), dropout=0, batchnorm = False):

        self.q_net      = NN(n_agents*(in_size + action_space_size), 1, layers=layers_q, final_activation=final_activation_q, activation=activation_q, dropout=dropout, batchnorm = batchnorm)
        self.policy_net = NN(in_size, action_space_size, layers=layers_p, final_activation=final_activation_p, activation=activation_p, dropout=dropout, batchnorm = batchnorm)

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
    def __init__(self, n_agents, dim_obs, dim_act , layers_p=[100], layers_q=[10, 10], lr_q=0.001, lr_policy=0.01, explo=0.1, memory_size=1000, batchnorm = True, **kwargs):

        self.network        = QPolicyNet_MA(n_agents=n_agents, in_size=dim_obs, action_space_size=dim_act, layers_p=layers_p, layers_q=layers_q, batchnorm=batchnorm)
        self.target_net     = deepcopy(self.network)
        self.optim_q        = torch.optim.Adam(params=self.network.q_net.parameters(), lr=lr_q)
        self.optim_policy   = torch.optim.Adam(params=self.network.policy_net.parameters(), lr=lr_policy)

        self.memory = Memory(mem_size=memory_size)
        self.noise = Orn_Uhlen(n_actions=dim_act, sigma=explo)

        self.loss = torch.nn.SmoothL1Loss() #torch.nn.MSELoss()
        self.test = False

    def store(self, transition):
        if not self.test:
            self.memory.store(transition)

class MADDPG():
    """Dans le cas du MADDPG, il faut utiliser un Q qui prend en compte les paramètres des autres DDPG
    pour une optimisation "collaborative" de tous les DDPG
    """
    def __init__(self, env, world, opt, layers_p=[100, 100], layers_q=[100,100], batch_per_learn=10, batch_size=1000, memory_size=1000000, lr_q=[0.001], lr_policy=[0.0005],
                 discount=0.99, rho=0.01, start_events=100000, explo=0.1, grad_clip_q = 0.5, grad_clip_policy=0.5, batchnorm=False, **kwargs):
        #super().__init__(env, opt) #seulement si on fait un héritage de agent pour obtenir le self.action_space et self.test

        self.n_agents = len(env.agents)
        if len(lr_q)!=self.n_agents:
            lr_q = [lr_q[-1]] * self.n_agents
            lr_policy = [lr_policy[-1]] * self.n_agents

        # Compute the dimensions of the env
        junk = env.reset()
        junk = [o.shape[0] for o in junk]
        self.obs_size = max(junk)
        self.action_size = world.dim_p

        self.agents = [DDPG_adapt(self.n_agents, self.obs_size, self.action_size , layers_p=layers_p, layers_q =layers_q, lr_q=lr_q[k], lr_policy=lr_policy[k], explo=explo, memory_size=memory_size, batchnorm=batchnorm) for k in range(len(lr_q))]

        self.config = opt
        self.featureExtractor = NothingToDo(env, self.obs_size) #opt.featExtractor(env)
        self.loss = torch.nn.SmoothL1Loss()
        self.p_lr = lr_policy #list
        self.q_lr = lr_q #list
        self.grad_clip_policy = grad_clip_policy
        self.grad_clip_q = grad_clip_q
        self.discount = discount
        self.explo = explo

        self.test = False
        self.batch_size = batch_size
        self.memory_size = memory_size
        self.batch_per_learn = batch_per_learn
        self.startEvents = start_events
        self.noise = Orn_Uhlen(n_actions=self.action_size, sigma=self.explo)

        self.rho = rho

        self.freq_optim = self.config.freqOptim
        self.n_events   = 0

        self.min_act = -1.
        self.max_act = 1.
        self.min_reward = -10. #ou -100
        self.max_reward = 10.

    def act(self, obs):
        for agent in self.agents :
            agent.network.policy_net.eval()

        if self.n_events < self.startEvents:
            # random actions to fill the buffer when starting
            actions = torch.randn(self.n_agents, self.action_size)
        else:
            with torch.no_grad():
                actions = torch.cat([agent.network.policy(obs[k].unsqueeze(0)) for k,agent in enumerate(self.agents)]).view(self.n_agents,self.action_size)
        n = torch.zeros(1)
        if not self.test:
            n = self.noise.sample()
            # n = torch.rand(1) * self.explo
        o = (actions + n) #.squeeze()
        return o #torch.clamp(o, self.min_act, self.max_act) #rescale au lieu de clamp

    def time_to_learn(self):
        self.n_events += 1
        if self.n_events % self.freq_optim != 0 or self.n_events < self.freq_optim or self.test:
            return False
        else:
            return True

    def learn(self, episode_done):
        mean_qloss, mean_ploss = [0]*self.n_agents, [0]*self.n_agents
        for k in range(self.batch_per_learn):
            qloss, ploss, q_mean = self._train_batch()
            mean_qloss = [mean_qloss[i]+qloss[i] for i in range(len(qloss))]
            mean_ploss = [mean_ploss[i]+ploss[i] for i in range(len(ploss))]
        return {'Q Loss - Agent '+str(i): mean_qloss[i]/self.batch_per_learn for i in range(len(mean_qloss))}\
               |{'Policy Loss - Agent '+str(i): mean_ploss[i]/self.batch_per_learn for i in range(len(mean_ploss))}\
               |{'Q mean - Agent '+str(i): q_mean[i] for i in range(len(q_mean))}

    def _train_batch(self):
        batches = [agent.memory.sample_batch(batch_size=self.batch_size) for agent in self.agents]

        # bs = batches[-1]['obs'].shape[0]
        b_obs = [batches[k]['obs'] for k in range(self.n_agents)]
        b_action = [batches[k]['action'] for k in range(self.n_agents)]
        b_reward = [batches[k]['reward'] for k in range(self.n_agents)]
        b_new = [batches[k]['new_obs'] for k in range(self.n_agents)]
        b_done = [batches[k]['done'] for k in range(self.n_agents)]

        # update q net
        q_loss, q_mean = self._update_q_all(b_obs, b_action, b_reward, b_new, b_done) #torch.cat(b_obs sur la 2e dim)

        # update policy
        loss_policy = self._update_policy_all(b_obs)

        # update target network
        self._update_target_all()

        return q_loss, loss_policy, q_mean

    def _update_q_all(self, b_obs, b_action, b_reward, b_new, b_done):
        """Gradient descent on Q-value function"""
        q_loss_all = []
        q_mean_all = []
        bs = b_obs[-1].shape[0]
        for i, agent in enumerate(self.agents):
            # switch agent net to train mode
            agent.network.q_net.train()
            # we start with computing the target
            with torch.no_grad():
                # print(b_new[i][:,i,:].shape)
                target_next_act = torch.cat([agent.target_net.policy(b_new[i][:,i,:]) for i,agent in enumerate(self.agents)], dim=-1)
                # print("target",target_next_act.shape)
                # print("b_act", b_action[i].view(bs,-1).shape)
                target = b_reward[i] + self.discount * agent.target_net.q(b_new[i].view(bs,-1), target_next_act) * (~b_done[i]).float()
            preds_q = agent.network.q(b_obs[i].view(bs,-1), b_action[i].view(bs,-1))
            q_mean_all.append(target.mean()) # preds_q.mean()

            q_loss = agent.loss(preds_q, target.detach())
            q_loss_all.append(q_loss)

            agent.optim_q.zero_grad()
            q_loss.backward()
            # to prevent from gradient exploding
            torch.nn.utils.clip_grad_norm_(agent.network.q_net.parameters(), self.grad_clip_q)
            agent.optim_q.step()

            # switch agent net to eval mode
            agent.network.q_net.eval()

        return q_loss_all, q_mean_all

    def _update_policy_all(self, b_obs):
        """Gradient ascent on the policy"""
        loss_policy_all = []
        bs = b_obs[-1].shape[0]
        for agent in self.agents :
            # switch to train mode
            agent.network.policy_net.train()
        for i, agent in enumerate(self.agents):
            # print("b_obs", b_obs[i][:,i,:].shape)
            with torch.no_grad():
                pred_act = torch.cat(
                    [agent.network.policy(b_obs[k][:,k,:]) for k,agent in enumerate(self.agents)], dim=-1)
            # print("pred_act policy", pred_act.shape)
            # preds_q = agent.network.q(b_obs[i].view(bs, -1), b_action[i].view(bs, -1))
            pred_q = agent.network.q(b_obs[i].view(bs, -1), pred_act)
            loss_policy = -torch.mean(pred_q)
            loss_policy_all.append(loss_policy)

            agent.optim_policy.zero_grad()
            loss_policy.backward()
            # to prevent from gradient explosion
            torch.nn.utils.clip_grad_norm_(agent.network.policy_net.parameters(), self.grad_clip_policy)
            agent.optim_policy.step()

        # back to eval mode
        for agent in self.agents:
            agent.network.policy_net.eval()

        return loss_policy_all

    def _update_target_all(self):
        """Soft update, using rho"""
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
