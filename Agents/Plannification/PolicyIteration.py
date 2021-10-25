from Agents.Plannification.PlannerAgent import PlannerAgent
from numpy import random
import numpy as np


class PolicyIterationAgent(PlannerAgent):

    def __init__(self, action_space, env):
        super().__init__(action_space, env)

    def compute_policy(self, epsilon, gamma):
        old_policy = []
        while old_policy != self.policy:
            old_policy = self.policy.copy()
            self.__policy_evaluation(old_policy, epsilon, gamma)
            self.__policy_improvement(gamma)

        self.ready = True
        return 0

    def __policy_evaluation(self, old_policy, epsilon, gamma):
        p = self._p
        r = self._r
        states = self.states

        self.V = [random.rand() for s in range(len(states))]

        while sum([abs(p - q) for p, q in zip(self.V, self.oldV)]) > epsilon:
            self.oldV = self.V.copy()
            for s in range(len(states)):
                vs = sum([p(s, old_policy[s], u) * (r(s, old_policy[s], u) + gamma * self.oldV[u]) for u in range(len(states))])
                self.V[s] = vs
        return 0

    def __policy_improvement(self, gamma):
        p = self._p
        r = self._r
        states = self.states

        for s in range(len(states)):
            self.policy[s] = np.argmax(
                [sum([p(s, a, u) * (r(s, a, u) + gamma * self.V[u]) for u in range(len(states))]) for a in
                 range(self.action_space.n)])

        return 0

    def act(self, observation):
        if not self.ready:
            print('Warning: policy has not been computed yet')
        index = self.states.index(str(observation.tolist()))
        return self.policy[index]