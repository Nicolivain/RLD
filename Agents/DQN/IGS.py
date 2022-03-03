import copy

import numpy as np
import torch
from Agents.DQN.DQNGoal import DQNGoal
from pytorch_lightning import seed_everything
from torch import nn
from torch.nn import functional
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from Structure.Memory import Memory
from Tools.exploration import pick_greedy, pick_epsilon_greedy, pick_ucb
from copy import deepcopy
from collections import deque, defaultdict
from torch.distributions import Categorical

class IGS(DQNGoal):
    def __init__(self, env, opt, layers, memory_size=10000, learning_rate=0.001, explo=0.1, explo_mode=0, discount=0.99, decay=0.9999, batch_size=1000, startEvents=2000, freq_update_target=1000, alpha = 0.3, beta = 0.9, HER = True, freq_feed_GoalBuffer = 10, **kwargs):
        super().__init__(env, opt, layers, memory_size, learning_rate, explo, explo_mode, discount, decay, batch_size, startEvents, freq_update_target, **kwargs)

        self.alpha = alpha # temperature for entropy
        self.beta = beta
        self.HER = HER # for IGS, one can add HER or not
        self.freq_feed_GoalBuffer = freq_feed_GoalBuffer

        # Goal buffers
        self.G = deque(maxlen=10)  # 10 is the number of fictive goals
        self.N = defaultdict(lambda: 0)  # number of tries for reaching the goal
        self.V = defaultdict(lambda: 0)  # number of times we actually reached the goal
        self.freq_feed_GoalBuffer = freq_feed_GoalBuffer

    def sample_artificial_goal(self):
        n = torch.tensor([self.N[str(g)] for g in self.G], dtype=float)
        v = torch.tensor([self.V[str(g)] for g in self.G], dtype=float)
        n = n.clamp(min=1.)
        ratio = v / n
        H = -ratio * torch.log(ratio + 1e-9) - (1 - ratio) * torch.log(1 - ratio + 1e-9)
        logits = torch.exp(self.alpha * H)
        law = Categorical(logits=logits)  # sample Bernoulli integers according to given logits
        idx = int(law.sample().item())
        return self.G[idx], H

    def goal_in_buffer(self, goal):
        for g in self.G:
            if (goal == g).all():
                return 1  # we do nothing
        self.G.append(goal)
        return 0