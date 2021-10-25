from numpy import random
import numpy as np
from abc import abstractmethod, ABC


class PlannerAgent(ABC):

    def __init__(self, action_space, env):
        self.action_space = action_space
        self.env = env

        states, mdp = self.env.getMDP()
        self.states = states
        self.mdp = mdp
        self.V = [random.rand() for s in range(len(states))]
        self.oldV = [np.inf for s in range(len(states))]

        self.policy = [random.randint(action_space.n) for a in range(len(states))]
        self.ready = False

    def _p(self, s, a, u):
        try:
            possible_outcomes = [o[1] for o in self.mdp[s][a]]
            if u in possible_outcomes:
                ind = possible_outcomes.index(u)
                return self.mdp[s][a][ind][0]
            else:
                return 0
        except KeyError:
            return 0

    def _r(self, s, a, u):
        try:
            possible_outcomes = [o[1] for o in self.mdp[s][a]]
            if u in possible_outcomes:
                ind = possible_outcomes.index(u)
                return self.mdp[s][a][ind][2]
            else:
                return 0
        except KeyError:
            return 0

    @abstractmethod
    def compute_policy(self, epsilon, gamma):
        pass

    @abstractmethod
    def act(self, obs):
        pass


class RandomAgent(PlannerAgent):
    """The world's simplest agent!"""

    def __init__(self, action_space, env):
        super().__init__(action_space, env)

    def compute_policy(self, epsilon, gamma):
        pass

    def act(self, observation):
        return self.action_space.sample()



