import argparse
import sys
import matplotlib
import gym
import Tools.gridworld
import torch
from Tools.utils import *
from Tools.core import *
from torch.utils.tensorboard import SummaryWriter
from matplotlib import pyplot as plt
import yaml
from datetime import datetime
from Agents.Policy.A2C import A2C

matplotlib.use("Qt5agg")
# matplotlib.use("TkAgg")

if __name__ == '__main__':

    mode = ['RandomAgent', 'A2C'][1]
    env, config, outdir, logger = init('Training/configs/config_random_cartpole.yaml', mode)

    torch.manual_seed(config['seed'])
    freqTest = config["freqTest"]
    freqSave = config["freqSave"]
    nbTest = config["nbTest"]
    env.seed(config["seed"])
    torch.manual_seed(config["seed"])
    episode_count = config["nbEpisodes"]

    agent = {
             'A2C' : A2C(env, config, layers=[30, 30], batch_size=1000, memory_size=1000)
             }[mode]

    rsum = 0
    mean = 0
    verbose = True
    itest = 0
    reward = 0
    done = False
    loss = -1
    for i in range(episode_count):
        checkConfUpdate(outdir, config)

        rsum = 0
        agent.nbEvents = 0
        ob = env.reset()

        # On souhaite afficher l'environnement (attention à ne pas trop afficher car çà ralentit beaucoup)
        if i % int(config["freqVerbose"]) == 0:
            verbose = True
        else:
            verbose = False

        # C'est le moment de tester l'agent
        if i % freqTest == 0 and i >= freqTest:  # Same as train for now
            print("Test time! ")
            mean = 0
            agent.test = True

        # On a fini cette session de test
        if i % freqTest == nbTest and i > freqTest:
            print("End of test, mean reward=", mean / nbTest)
            itest += 1
            logger.direct_write("rewardTest", mean / nbTest, itest)
            agent.test = False

        # C'est le moment de sauver le modèle
        if i % freqSave == 0:
            agent.save(outdir + "/save_" + str(i))

        j = 0
        if verbose:
            env.render()

        new_ob = agent.featureExtractor.getFeatures(ob)
        while True:
            if verbose:
                env.render()

            ob = torch.from_numpy(new_ob)
            action = agent.act(ob)
            new_ob, reward, done, _ = env.step(action)
            new_ob = agent.featureExtractor.getFeatures(new_ob)

            j += 1

            # Si on a atteint la longueur max définie dans le fichier de config
            if ((config["maxLengthTrain"] > 0) and (not agent.test) and (j == config["maxLengthTrain"])) or (agent.test and (config["maxLengthTest"] > 0) and (j == config["maxLengthTest"])):
                done = True
                print("forced done!")

            transition = {
                'obs': ob,
                'action': action,
                'new_state': torch.from_numpy(new_ob),
                'reward': reward,
                'done': done,
                'it': j
            }
            agent.store(transition)
            rsum += reward

            if agent.time_to_learn():
                loss = agent.learn(done)
            if done and loss > 0:
                print(str(i) + " rsum=" + str(rsum) + ", " + str(j) + " actions " + f' loss: {loss}')
                logger.direct_write("reward", rsum, i)
                logger.direct_write('loss', loss, i)
                agent.nbEvents = 0
                mean += rsum
                rsum = 0
            if done:
                break

    env.close()
