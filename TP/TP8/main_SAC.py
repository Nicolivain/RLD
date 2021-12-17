import matplotlib
import torch
import numpy
from Agents.Continuous.SAC import SAC
from Tools.core import *
from Tools.utils import *

matplotlib.use("Qt5agg")
# matplotlib.use("TkAgg")

if __name__ == '__main__':

    torch.manual_seed(1)
    numpy.random.seed(1)

    agent = SAC()

    env = gym.make('Pendulum-v1')

    score = 0.0
    print_interval = 20

    for n_epi in range(10000):
        s = env.reset()
        done = False

        while not done:
            a = agent.act(torch.from_numpy(s).float())
            s_prime, r, done, info = env.step([2.0 * a])
            transition = {
                'obs': s,
                'action': a,
                'reward': r / 10.0,
                'new_obs': s_prime,
                'done': done
            }
            agent.memory.put(transition)
            score += r
            s = s_prime

        if agent.memory.nentities > 1000:
            agent.learn(done)

        if n_epi % print_interval == 0 and n_epi != 0:
            print("# of episode :{}, avg score : {:.1f}".format(n_epi, score / print_interval))
            score = 0.0

    env.close()
