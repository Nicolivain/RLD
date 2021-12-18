import matplotlib

from Agents.DQN.DQN import DQN
from Agents.DQN.DuelingDQN import DuelingDQN
from Agents.DQN.ReplayDQN import ReplayDQN
from Agents.DQN.TargetDQN import TargetDQN
from Agents.DQN.minDQN import MinDQN
from Tools.core import *
from Tools.utils import *
import Tools.gridworld

# matplotlib.use("Qt5agg")
matplotlib.use("TkAgg")

if __name__ == '__main__':

    mode = ['DQN', 'ReplayDQN', 'TargetDQN', 'minDQN', 'DuelingDQN'][4]
    env, config, outdir, logger = init('Config/env_config/config_random_lunar.yaml', mode)

    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    freqTest = config["freqTest"]
    freqSave = config["freqSave"]
    nbTest = config["nbTest"]
    env.seed(config["seed"])
    np.random.seed(config["seed"])
    episode_count = config["nbEpisodes"]

    agent = {'DQN'          : DQN(env, config, layers=[200]),
             'ReplayDQN'    : ReplayDQN(env, config, layers=[24, 24], memory_size=3000, batch_size=64),
             'TargetDQN'    : TargetDQN(env, config, layers=[24, 24], update_target=500),
             'minDQN'       : MinDQN(env, config, layers=[24, 24], memory_size=10000, batch_size=256, update_target=100),
             'DuelingDQN'   : DuelingDQN(env, config, layers=[24, 24], memory_size=3000, batch_size=64, update_target=500)
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

            ob = torch.from_numpy(new_ob).float()
            action = agent.act(ob)
            new_ob, reward, done, _ = env.step(action)
            new_ob = agent.featureExtractor.getFeatures(new_ob)

            j += 1

            # Si on a atteint la longueur max définie dans le fichier de config
            if ((config["maxLengthTrain"] > 0) and (not agent.test) and (j == config["maxLengthTrain"])) or (agent.test and (config["maxLengthTest"] > 0) and (j == config["maxLengthTest"])):
                done = True
                print("forced done!")

            # select what to save

            transition = {
                'obs'       : ob,
                'action'    : action,
                'reward'    : reward,
                'new_obs'   : torch.from_numpy(new_ob).float(),
                'done'      : done,
                'it'        : j
            }

            agent.store(transition)
            rsum += reward

            if agent.time_to_learn():
                results = agent.learn(done)
                loss = results['Loss']

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
