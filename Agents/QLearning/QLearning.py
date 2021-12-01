from Agents.QLearning.QAgent import QAgent


class QLearning(QAgent):

    def __init__(self, env, config):
        super().__init__(env, config)

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

        if done:
            self.explo *= self.decay

        return {}

    def act(self, obs):
        return super().act(obs)
