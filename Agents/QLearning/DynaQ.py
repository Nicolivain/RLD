from Agents.QLearning.QAgent import QAgent
from Structure.MDP import MDP


class DynaQ(QAgent):

    def __init__(self, env, config, planning_iter=100):
        super(DynaQ, self).__init__(env, config)
        self.mdp = MDP(alpha=self.alpha)
        self.planning_iter = planning_iter

    def act(self, obs):
        return super().act(obs)

    def learn(self, done):
        st = self.last_source
        rt = self.last_reward
        next_st = self.last_dest
        at = self.last_action
        if not self.sarsa:
            self.values[st][at] += self.alpha * (rt + self.discount * max(self.values[next_st]) - self.values[st][at])
        else:
            next_at = self.act(next_st)
            self.values[st][at] += self.alpha * (rt + self.discount * self.values[next_st][next_at] - self.values[st][at])

        self.mdp.update(st, at, next_st, rt)
        self.planning()

        if done:
            self.explo *= self.decay

        return {}

    def planning(self):
        for k in range(self.planning_iter):
            st, at         = self.mdp.sample()
            next_st, rt    = self.mdp.step(st, at)
            self.values[st][at] += self.alpha * (rt + self.discount * max(self.values[next_st]) - self.values[st][at])
