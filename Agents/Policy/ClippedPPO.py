from Agents.Agent import Agent


class ClippedPPO(Agent):
    def __init__(self, env, opt):
        super(ClippedPPO, self).__init__(env, opt)

    def act(self, obs):
        pass

    def learn(self, done):
        pass
