import torch


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
        while not done:
            a = self.agent.act(torch.from_numpy(s).float())
            played_a = self._scale_action(a)
            s_prime, r, done, info = self.env.step(played_a)
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
        if type(a) == int or type(a) == float:
            return [self.action_rescale * a]
        else:
            return [c * self.action_rescale for c in a]

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

        return mean

    def train_agent(self, outdir):
        itest    = 0
        mean     = 0
        res_dict = {}
        for i in range(self.env_config.nbEpisodes):

            # Initialize environnement
            rsum = 0
            s = self.env.reset()

            # Check if we test or if we display
            mean = self._test_agent_setup(i, mean, itest)
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
            print('Episode {:5d} Reward: {:3.1f} #Action: {:4d}'.format(i, rsum, n_action))

        print('Done')
