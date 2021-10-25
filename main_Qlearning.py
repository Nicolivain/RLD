import matplotlib
import numpy as np
from Agents.QLearning.DynaQ import DynaQ
from Agents.QLearning.QLearning import QLearning
from Tools.utils import *
import Tools.gridworld

matplotlib.use("TkAgg")
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


if __name__ == '__main__':

    env, config, outdir, logger = init('Training/configs/config_qlearning_gridworld.yaml', "QLearning")

    # Config
    freqTest = config["freqTest"]
    freqSave = config["freqSave"]
    nbTest = config["nbTest"]
    env.seed(config["seed"])
    np.random.seed(config["seed"])
    episode_count = config["nbEpisodes"]

    # Choose Agent
    agent = [QLearning(env, config),
             DynaQ(env, config)][1]

    agent.sarsa = 0

    rsum = 0
    mean = 0
    verbose = True
    itest = 0
    reward = 0
    done = False
    nb = 0

    test_rewards = []

    for i in range(episode_count):
        checkConfUpdate(outdir, config)  # permet de changer la config en cours de run
        rsum = 0
        agent.nbEvents = 0
        ob = env.reset()

        if i > 0 and i % int(config["freqVerbose"]) == 0:
            verbose = True
        else:
            verbose = False

        if i % freqTest == 0 and i >= freqTest:  # Si agent.test alors retirer l'exploration
            print("Test time! ")
            mean = 0
            agent.test = True

        if i % freqTest == nbTest and i > freqTest:
            print("End of test, mean reward=", mean / nbTest)
            itest += 1
            logger.direct_write("rewardTest", mean / nbTest, itest)
            test_rewards.append(mean/nbTest)
            agent.test = False

        if i % freqSave == 0:
            agent.save(outdir + "/save_" + str(i) + '.pkl')

        j = 0
        if verbose:
            env.render()
        new_ob = agent.store_state(ob)
        while True:
            if verbose:
                env.render()

            ob = new_ob
            action = agent.act(ob)
            new_ob, reward, done, _ = env.step(action)
            new_ob = agent.store_state(new_ob)

            j += 1

            if ((config["maxLengthTrain"] > 0) and (not agent.test) and (j == config["maxLengthTrain"])) or (agent.test and (config["maxLengthTest"] > 0) and (j == config["maxLengthTest"])):
                done = True
                # print("forced done!")

            agent.store(ob, action, new_ob, reward, done, j)
            agent.learn(done)
            rsum += reward
            if done:
                print(str(i) + " rsum=" + str(rsum) + ", " + str(j) + " actions ")
                logger.direct_write("reward", rsum, i)
                mean += rsum
                break

    env.close()
