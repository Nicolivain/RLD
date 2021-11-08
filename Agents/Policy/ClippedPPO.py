from Agents.Agent import Agent


class ClippedPPO(Agent):
    def __init__(self, env, opt, layers, k, delta=1.5, loss='smoothL1', memory_size=1000, batch_size=1000):
        super(ClippedPPO, self).__init__(env, opt)

    def act(self, obs):
        pass

    def learn(self, done):
        pass
