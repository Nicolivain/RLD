import numpy as np
from numpy import random


def pick_greedy(values):
    return np.argmax(values)


def pick_epsilon_greedy(values, epsilon):
    assert len(values.shape) == 1, 'Values must be unidimensional'
    if random.random() < epsilon:
        return random.randint(len(values))
    else:
        return pick_greedy(values)


def pick_ucb(values):
    pass  # TODO
