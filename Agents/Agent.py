from abc import abstractmethod, ABC
import pickle


class Agent(ABC):

    def __init__(self, env, config):
        self.env            = env
        self.config         = config
        self.action_space   = env.action_space

        self.discount       = config.gamma
        self.decay          = config.decay
        self.alpha          = config.learningRate
        self.explo          = config.explo
        self.exploMode      = config.exploMode  # 0: epsilon greedy, 1: ucb

        self.test           = False

    @abstractmethod
    def act(self, obs):
        pass

    @abstractmethod
    def learn(self, obs):
        pass

    def save(self, path):
        """
        Unknown bug: can't pickle NoneType ??
        f = open(path, 'wb')
        pickle.dump(self, f)
        f.close()
        """
        pass
