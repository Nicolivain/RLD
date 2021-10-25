import argparse
import sys
import matplotlib
import gym
import gridworld
import torch
from utils import *
from core import *
from torch.utils.tensorboard import SummaryWriter
from matplotlib import pyplot as plt
import yaml
from datetime import datetime
from Agents import *

matplotlib.use("Qt5agg")
# matplotlib.use("TkAgg")

if __name__ == '__main__':

    mode = ['RandomAgent', 'DQN', 'ReplayDQN', 'TargetDQN'][3]
    env, config, outdir, logger = init('./configs/config_random_cartpole.yaml', mode)

    torch.manual_seed(config['seed'])
    freqTest = config["freqTest"]
    freqSave = config["freqSave"]
    nbTest = config["nbTest"]
    env.seed(config["seed"])
    np.random.seed(config["seed"])
    episode_count = config["nbEpisodes"]

    agent = {'RandomAgent'  : RandomAgent(env, config),
             'DQN'          : VanillaDQN(env, config, train_batch_size=1),
             'ReplayDQN'    : ReplayDQN(env, config, replay_size=10000, batch_size=100),
             'TargetDQN'    : TargetNetworkDQN(env, config, replay_size=10000, batch_size=100, target_update=1000)}[mode]

    rsum = 0
    mean = 0
    verbose = True
    itest = 0
    reward = 0
    done = False
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

            ob = new_ob
            action = agent.act(ob)
            new_ob, reward, done, _ = env.step(action)
            new_ob = agent.featureExtractor.getFeatures(new_ob)

            j += 1

            # Si on a atteint la longueur max définie dans le fichier de config
            if ((config["maxLengthTrain"] > 0) and (not agent.test) and (j == config["maxLengthTrain"])) or (agent.test and (config["maxLengthTest"] > 0) and (j == config["maxLengthTest"])):
                done = True
                print("forced done!")

            agent.store(ob, action, new_ob, reward, done, j)
            rsum += reward

            if agent.timeToLearn(done):
                agent.learn(done)
            if done:
                print(str(i) + " rsum=" + str(rsum) + ", " + str(j) + " actions ")
                logger.direct_write("reward", rsum, i)
                agent.nbEvents = 0
                mean += rsum
                rsum = 0

                break

    env.close()
