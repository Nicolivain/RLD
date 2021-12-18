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

    env = gym.make('Pendulum-v1')
    agent = SAC(env, load_yaml('Training/configs/config_random_pendulum.yaml'))

    score = 0.0
    print_interval = 20

    for n_epi in range(10000):
        s = env.reset()
        done = False

        while not done:
            a, log = agent.policy(torch.from_numpy(s).float())
            s_prime, r, done, info = env.step([2.0 * a.item()])
            transition = {
                'obs': s,
                'action': a.item(),
                'reward': r / 10.0,
                'new_obs': s_prime,
                'done': done
            }
            agent.memory.put(transition)
            score += r
            s = s_prime

        if agent.time_to_learn(done):
            agent.learn(done)

        if n_epi % print_interval == 0 and n_epi != 0:
            if n_epi % print_interval == 0 and n_epi != 0:
                print("# of episode :{}, avg score : {:.1f} alpha:{:.4f}".format(n_epi, score / print_interval,
                                                                                 agent.log_alpha.exp()))
            score = 0.0

    env.close()
