from abc import abstractmethod, ABC


class Agent(ABC):

    def __init__(self, env, config):
        self.env            = env
        self.config         = config
        self.action_space   = env.action_space

        self.featureExtractor = config.featExtractor(env)

        self.test           = False

    @abstractmethod
    def act(self, obs):
        pass

    @abstractmethod
    def learn(self, done):
        pass

    def store(self, transition):
        pass

    def time_to_learn(self):
        return True

    def save(self, path):
        pass
