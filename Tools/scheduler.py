import datetime
from copy import deepcopy

import torch
from numpy.random import choice

from Tools.utils import *


class TaskScheduler:
    def __init__(self, agent, search_space, config_path, agent_params, save_path, model_save_tag=''):

        self.agent          = agent
        self.search_space   = search_space
        self.config_path    = config_path
        self.agent_params   = agent_params
        self.save_path      = save_path
        self.model_tag      = model_save_tag

    def randomize_args(self, args):
        for key, possible_values in self.search_space.items():
            if key in args:
                args[key] = choice(possible_values, 1)[0]
                if type(args[key]) == list or type(args[key]) == tuple:
                    continue
                elif args[key] / int(args[key]) == 1.0:
                    args[key] = int(args[key])
                else:
                    args[key] = float(args[key])
        return args

    def search(self, n_test):
        for i in range(n_test):
            print(f'Training model: {i + 1}')
            start_time = datetime.now().strftime("%Y%m%d-%H%M%S")
            dir_path = os.path.join(self.save_path, 'tag-' + start_time + '_' + self.model_tag)

            env, config, outdir, logger = init(self.config_path, self.model_tag, outdir=dir_path, copy_config=False, launch_tb=False)

            # manual seed for reproduciability
            torch.manual_seed(config['seed'])

            agent_params = deepcopy(self.agent_params)

            # random search
            agent_params = self.randomize_args(agent_params)
            config       = self.randomize_args(config)
            write_yaml(os.path.join(dir_path, 'agent_params'), agent_params)
            write_yaml(os.path.join(dir_path, 'config'), config)
            print(agent_params)
            print(config)
            # adding environnement parameters
            agent_params['env'] = env
            agent_params['opt'] = config

            agent = self.agent(**agent_params)
            TaskScheduler.train_agent(agent, env, config, outdir, logger)
            del agent, agent_params

    @staticmethod
    def train_agent(agent, env, config, outdir, logger, render_env=False):

        env.seed(config["seed"])

        freqTest      = config["freqTest"]
        freqSave      = config["freqSave"]
        nbTest        = config["nbTest"]
        episode_count = config["nbEpisodes"]

        mean = 0
        itest = 0
        policy_loss, value_loss = None, None
        for i in range(episode_count):
            rsum = 0
            agent.nbEvents = 0
            ob = env.reset()

            # On souhaite afficher l'environnement (attention à ne pas trop afficher car çà ralentit beaucoup)
            if i % int(config["freqVerbose"]) == 0 and render_env:
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
                if ((config["maxLengthTrain"] > 0) and (not agent.test) and (j == config["maxLengthTrain"])) or (
                        agent.test and (config["maxLengthTest"] > 0) and (j == config["maxLengthTest"])):
                    done = True
                    print("forced done!")

                transition = {
                    'obs': ob,
                    'action': action,
                    'new_obs': torch.from_numpy(new_ob),
                    'reward': reward / 100,  # rescale factor for NN
                    'done': done,
                    'it': j
                }
                agent.store(transition)
                rsum += reward

                if agent.time_to_learn():
                    value_loss, policy_loss = agent.learn(done)
                if done and policy_loss is not None:
                    print(
                        'Episode {:5d} Reward: {:3.1f} #Action: {:4d} Policy Loss: {:1.6f} Value Loss: {:1.6f}'.format(
                            i, rsum, j, policy_loss, value_loss))
                    logger.direct_write("reward", rsum, i)
                    logger.direct_write('average policy loss', policy_loss, i)
                    logger.direct_write('value loss', value_loss, i)
                    agent.nbEvents = 0
                    mean += rsum
                    rsum = 0
                if done:
                    break

        env.close()
        print('done\n\n')


