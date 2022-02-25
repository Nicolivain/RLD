import torch
import numpy as np


class Trainer:
    def __init__(self, agent, env, env_config, agent_params, logger, reward_rescale=1, action_rescale=1):

        self.agent  = agent(**agent_params)
        self.env    = env
        self.logger = logger

        self.env_config     = env_config
        self.reward_rescale = reward_rescale
        self.action_rescale = action_rescale

    def run_episode(self, s):
        s = self.agent.featureExtractor.getFeatures(s)
        done = False
        score = 0
        n_action = 0
        while not done and n_action < self.env_config.maxLengthTrain:
            a = self.agent.act(torch.from_numpy(s).float())
            played_a = self._scale_action(a)
            s_prime, r, done, info = self.env.step(played_a.copy())
            s_prime = self.agent.featureExtractor.getFeatures(s_prime)
            transition = {
                'obs': s,
                'action': a,
                'reward': r / self.reward_rescale,
                'new_obs': s_prime,
                'done': done
            }
            self.agent.store(transition)
            score += r
            n_action += 1
            s = s_prime

        return score, n_action

    def _scale_action(self, a):
        if type(a) == list:
            return [self.action_rescale * c for c in a]
        elif type(a) == torch.Tensor:
            return self.action_rescale * a.numpy()
        else:
            return self.action_rescale * a

    def _display_setup(self, i):
        # On souhaite afficher l'environnement (attention à ne pas trop afficher car çà ralentit beaucoup)
        if i % int(self.env_config["freqVerbose"]) == 0:
            verbose = True
        else:
            verbose = False

        if verbose:
            self.env.render()

    def _test_agent_setup(self, i, mean, itest):
        # C'est le moment de tester l'agent
        if i % self.env_config.freqTest == 0 and i >= self.env_config.freqTest:  # Same as train for now
            print("Test time! ")
            self.agent.test = True
            mean = 0

        # On a fini cette session de test
        if i % self.env_config.freqTest == self.env_config.nbTest and i > self.env_config.freqTest:
            print("End of test, mean reward = ", mean / self.env_config.nbTest)
            itest += 1
            self.logger.direct_write("Test Reward", mean / self.env_config.nbTest, itest)
            self.agent.test = False

        return mean, itest

    def train_agent(self, outdir, print_every=1):
        itest    = 0
        mean     = 0
        res_dict = {}
        for i in range(self.env_config.nbEpisodes):

            # Initialize environnement
            rsum = 0
            s = self.env.reset()

            # Check if we test or if we display
            mean, itest = self._test_agent_setup(i, mean, itest)
            self._display_setup(i)

            # Check if we save
            if i % self.env_config.freqSave == 0:
                self.agent.save(outdir + "/save_" + str(i))

            # Runing the episode and training
            rsum, n_action = self.run_episode(s)
            if self.agent.time_to_learn():
                res_dict = self.agent.learn(True)

            # Logging
            self.logger.direct_write("Reward", rsum, i)
            for k, v in res_dict.items():
                self.logger.direct_write(k, v, i)

            mean += rsum
            if i % print_every == 0:
                print('Episode {:5d} Reward: {:3.1f} #Action: {:4d}'.format(i, rsum, n_action))

        print('Done')

class Trainer_MADDPG:
    def __init__(self, agent, env, env_config, agent_params, logger, reward_rescale=1, action_rescale=1):

        self.agent  = agent(**agent_params)
        self.env    = env
        self.logger = logger

        self.env_config     = env_config
        self.reward_rescale = reward_rescale
        self.action_rescale = action_rescale

    def run_episode(self, s):
        s = self.agent.featureExtractor.getFeatures(s)
        done = [False]
        score = 0
        n_action = 0
        while not sum(done) and n_action < self.env_config.maxLengthTrain:
            a = self.agent.act(torch.from_numpy(s).float())
            played_a = self._scale_action(a)
            s_prime, r, done, info = self.env.step(played_a.copy())  #be careful : only play with a copy of the wanted action (step modifies its argument)
            s_prime = self.agent.featureExtractor.getFeatures(s_prime)
            transition = {
                'obs': s,
                'action': a,
                'reward': np.clip(r[0], self.agent.min_reward, self.agent.max_reward)*self.reward_rescale, #rewards are common, thus we can just save the first coordinate
                'new_obs': s_prime,
                'done': done
            }
            self.agent.store(transition)
            score = r[0]*self.reward_rescale
            n_action += 1
            s = s_prime

        return score, n_action

    def _scale_action(self, a):
        if type(a) == list:
            return [self.action_rescale * c for c in a]
        elif type(a) == torch.Tensor:
            return self.action_rescale * a.numpy()
        else:
            return self.action_rescale * a

    def _display_setup(self, i):
        # On souhaite afficher l'environnement (attention à ne pas trop afficher car çà ralentit beaucoup)
        if i % int(self.env_config["freqVerbose"]) == 0:
            verbose = True
        else:
            verbose = False

        if verbose:
            self.env.render()

    def _test_agent_setup(self, i, mean, itest):
        # C'est le moment de tester l'agent
        if i % self.env_config.freqTest == 0 and i >= self.env_config.freqTest:  # Same as train for now
            print("Test time! ")
            self.agent.test = True
            mean = 0

        # On a fini cette session de test
        if i % self.env_config.freqTest == self.env_config.nbTest and i > self.env_config.freqTest:
            print("End of test, common reward: ", mean)
            itest += 1
            self.logger.direct_write("Test Reward ", mean / self.env_config.nbTest, itest)
            self.agent.test = False

        return mean, itest

    def train_agent(self, outdir, print_every=1):
        itest    = 0
        mean     = 0
        res_dict = {}
        for i in range(self.env_config.nbEpisodes):

            # Initialize environnement
            s = self.env.reset()

            # Check if we test or if we display
            mean, itest = self._test_agent_setup(i, mean, itest)
            self._display_setup(i)

            # Check if we save
            if i % self.env_config.freqSave == 0:
                self.agent.save(outdir + "/save_" + str(i))

            # Running the episode and training
            rsum, n_action = self.run_episode(s)
            if self.agent.time_to_learn():
                res_dict = self.agent.learn(True)

            # Logging
            self.logger.direct_write("Common Rewards", rsum, i)
            if i % self.env_config["freqOptim"] == 0 :
                for k, v in res_dict.items():
                    self.logger.direct_write(k, v, i//self.env_config["freqOptim"]) #add modulo to plot only when optimization occurs

            mean += rsum
            if i % print_every == 0:
                print('Episode {:5d} \t Total reward: {:3.1f} \t #Action: {:4d}'.format(i, rsum, n_action))

        print('Done')
