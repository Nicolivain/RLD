from numpy.random import random, choice


class MDP:
    """
    A MDP model built to be updated and constructed during the learning phase
    """

    def __init__(self, alpha):
        self.transitions = {}
        self.proba       = {}
        self.rewards     = {}
        self.alpha       = alpha

    def __refresh(self, s, a, next_s, r):
        """
        applies the formula to the rewards and probabilities
        """

        self.rewards[s][a][next_s]  += self.alpha * (r - self.proba[s][a][next_s])
        self.proba[s][a][next_s] += self.alpha * (1 - self.proba[s][a][next_s])
        for other_s in self.transitions[s][a]:
            if other_s == next_s:
                continue
            self.proba[s][a][other_s] -= self.alpha * self.proba[s][a][other_s]

    def update(self, s, a, next_s, r):
        """
        updates the model with the new transition
        """
        if s not in self.transitions.keys():
            # si on ne connais pas cet état
            self.transitions[s] = {a: [next_s]}
            self.proba[s]       = {a: {next_s: 1}}
            self.rewards[s]     = {a: {next_s: r}}
        elif a not in self.transitions[s]:
            # si on ne connais pas cette action pour cet état
            self.transitions[s][a]  = [next_s]
            self.proba[s][a]        = {next_s: 1}
            self.rewards[s][a]      = {next_s: r}
        elif next_s not in self.transitions[s][a]:
            # si on ne connais pas cette résultante pour ce couple état-action
            self.transitions[s][a].append(next_s)
            self.proba[s][a][next_s]    = 0
            self.rewards[s][a][next_s]  = r

        self.__refresh(s, a, next_s, r)

    def step(self, s, a):
        """
        returns one possible outcome for state s and action a according to the learnt probailities
        """
        if s not in self.transitions or a not in self.transitions[s]:
            raise ValueError('State-Action couple unknown')
        rand = random()
        options = self.proba[s][a]
        cdf = 0

        for next_s, p in options.items():
            cdf += p
            if rand < cdf:
                return next_s, self.rewards[s][a][next_s]
        raise ValueError('Sum of probability is not 1')

    def sample(self):
        s = choice(list(self.transitions.keys()), 1)[0]
        a = choice(list(self.transitions[s].keys()), 1)[0]
        return s, a


