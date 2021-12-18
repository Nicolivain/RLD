import matplotlib

from Agents.Continuous.DDPG import DDPG
from Tools.core import *
from Tools.utils import *
from Core.Trainer import Trainer

matplotlib.use("Qt5agg")
# matplotlib.use("TkAgg")

if __name__ == '__main__':
    env, config, outdir, logger = init('Config/env_config/config_random_mountain_car.yaml', 'DDPG', outdir=None,
                                       copy_config=False, launch_tb=False)
    agent = DDPG

    xp = Trainer(agent=agent,
                 env=env,
                 env_config=config,
                 agent_params={'env': env, 'opt': config},
                 logger=logger,
                 reward_rescale=10,
                 action_rescale=2.0)

    xp.train_agent(outdir)


    """
    mode = ['DDPG'][0]
    env, config, outdir, logger = init('Config/env_config/config_random_mountain_car.yaml', mode)
    torch.manual_seed(config['seed'])
    freqTest = config["freqTest"]
    freqSave = config["freqSave"]
    nbTest = config["nbTest"]
    env.seed(config["seed"])
    torch.manual_seed(config["seed"])
    episode_count = config["nbEpisodes"]

    agent = {
             'DDPG'          : DDPG(env, config, batch_per_learn=3, layers=[100, 100], memory_size=1000000, batch_size=1024),
             }[mode]

    rsum = 0
    mean = 0
    verbose = True
    itest = 0
    reward = 0
    done = False
    policy_loss, value_loss = None, None
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
            new_ob, reward, done, _ = env.step(action.numpy())  # passage en numpy ici pour la compatibilité gym
            new_ob = agent.featureExtractor.getFeatures(new_ob)

            j += 1

            # Si on a atteint la longueur max définie dans le fichier de config
            if ((config["maxLengthTrain"] > 0) and (not agent.test) and (j == config["maxLengthTrain"])) or (agent.test and (config["maxLengthTest"] > 0) and (j == config["maxLengthTest"])):
                done = True
                print("forced done!")

            transition = {
                'obs': ob,
                'action': action,
                'new_obs': torch.from_numpy(new_ob),
                'reward': reward/100    ,   # rescale factor for NN
                'done': done,
                'it': j
            }
            agent.store(transition)
            rsum += reward

            if agent.time_to_learn():
                results = agent.learn(done)
                value_loss, policy_loss = results['Q Loss'], results['Policy Loss']
            if done and policy_loss is not None:
                # print(
                #     'Episode {:5d} Reward: {:3.1f} #Action: {:4d} Policy Loss: {:1.6f} Value Loss: {:1.6f}'.format(
                #         i, rsum, j, policy_loss, value_loss))
                logger.direct_write("reward", rsum, i)
                logger.direct_write('average policy loss', policy_loss, i)
                logger.direct_write('value loss', value_loss, i)
                agent.nbEvents = 0
                mean += rsum
                rsum = 0
            if done:
                break

    env.close()
    """
