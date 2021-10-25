from Agents.Plannification.PlannerAgent import PlannerAgent
import numpy as np


class ValueIterationAgent(PlannerAgent):
    """Value iteration algorithm"""

    def __init__(self, action_space, env):
        super().__init__(action_space, env)

    def compute_policy(self, epsilon, gamma):

        p = self._p
        r = self._r
        states = self.states

        while sum([abs(p - q) for p, q in zip(self.V, self.oldV)]) > epsilon:
            self.oldV = self.V.copy()
            for s in range(len(states)):
                options = [sum([p(s, a, u) * (r(s, a, u) + gamma * self.oldV[u]) for u in range(len(states))]) for a in
                           range(self.action_space.n)]
                self.V[s] = max(options)

        for s in range(len(states)):
            self.policy[s] = np.argmax(
                [sum([p(s, a, u) * (r(s, a, u) + gamma * self.V[u]) for u in range(len(states))]) for a in
                 range(self.action_space.n)])
        self.ready = True
        return 0

    def act(self, observation):
        if not self.ready:
            print('Warning: policy has not been computed yet')
        index = self.states.index(str(observation.tolist()))
        return self.policy[index]