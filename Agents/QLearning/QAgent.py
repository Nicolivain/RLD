import numpy as np
from Agents.Agent import Agent
from Tools.exploration import pick_greedy, pick_epsilon_greedy, pick_ucb


class QAgent(Agent):

    def __init__(self, env, config):
        super().__init__(env, config)

        self.qstates        = {}  # dictionnaire d'états rencontrés
        self.values         = []  # contient, pour chaque numéro d'état, les qvaleurs des self.action_space.n actions possibles

        self.last_source    = None
        self.last_action    = None
        self.last_dest      = None
        self.last_reward    = None
        self.last_done      = None

        self.sarsa          = config.sarsa

    def act(self, obs):
        values = self.values[obs]
        if self.test:
            return pick_greedy(values)
        else:
            if self.exploMode == 0:
                return pick_epsilon_greedy(values, self.explo)
            elif self.exploMode == 1:
                return pick_ucb(values)
            else:
                raise NotImplementedError(f'{self.exploMode} does not correspond to any available exploration function')

    def learn(self, done):
        raise NotImplementedError('Learning method requires override')

    def store_state(self, obs):
        observation = obs.dumps()
        s = str(observation)
        ss = self.qstates.get(s, -1)

        # Si l'etat jamais rencontré
        if ss < 0:
            ss = len(self.values)
            self.qstates[s] = ss
            self.values.append(np.ones(self.action_space.n) * 1.0)  # Optimism faced to uncertainty (on commence avec des valeurs à 1 pour favoriser l'exploration)
        return ss

    def store(self, ob, action, new_ob, reward, done, it):

        if self.test:
            return
        self.last_source = ob
        self.last_action = action
        self.last_dest = new_ob
        self.last_reward = reward
        if it == self.config.maxLengthTrain:   # si on a atteint la taille limite, ce n'est pas un vrai done de l'environnement
            done = False
        self.last_done = done
