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

    env, config, outdir, logger = init('Training/configs/config_random_pendulum.yaml', 'SAC', outdir=None, copy_config=False, launch_tb=False)
    agent = SAC(env, config)

    freqTest = config["freqTest"]
    freqSave = config["freqSave"]
    nbTest = config["nbTest"]

    score = 0.0
    print_interval = 20
    itest = 0
    mean = 0
    for n_epi in range(10000):
        s = env.reset()
        done = False

        # C'est le moment de tester l'agent
        if n_epi % freqTest == 0 and n_epi >= freqTest:  # Same as train for now
            print("Test time! ")
            itest = 0
            mean = 0
            agent.test = True

        # On a fini cette session de test
        if n_epi % freqTest == nbTest and n_epi > freqTest:
            # print("End of test, mean reward=", mean / nbTest)
            itest += 1
            logger.direct_write("Reward Test", mean / nbTest, itest)
            agent.test = False

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
            result_dict = agent.learn(done)
            for k, v in result_dict.items():
                logger.direct_write(k, v, n_epi)

        if n_epi % print_interval == 0 and n_epi != 0:
            if n_epi % print_interval == 0 and n_epi != 0:
                print("# of episode :{}, avg score : {:.1f} alpha:{:.4f}".format(n_epi, score / print_interval,
                                                                                 agent.log_alpha.exp()))
            score = 0.0

    env.close()
