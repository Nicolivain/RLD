import datetime
from copy import deepcopy

import torch
import yaml
from numpy.random import choice

from Tools.utils import *
from Core.Trainer import Trainer
import Tools.gridworld


class TaskScheduler:
    def __init__(self, agent, search_space, config_path, agent_config_path, save_path, model_save_tag=''):

        self.agent          = agent
        self.search_space   = search_space
        self.config_path    = config_path
        self.save_path      = save_path
        self.model_tag      = model_save_tag

        with open(agent_config_path) as f:
            self.agent_params = yaml.safe_load(f)

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

    def search(self, n_test, freq_print_episode=500, reward_rescale=100, action_rescale=1, render_env=False):
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
            write_yaml(os.path.join(dir_path, 'agent_params.yaml'), agent_params)
            write_yaml(os.path.join(dir_path, 'config.yaml'), config)
            print(agent_params)
            print(config)
            # adding environnement parameters
            agent_params['env'] = env
            agent_params['opt'] = config

            TaskScheduler.train_agent(self.agent, env, config, agent_params, outdir, logger, reward_rescale, action_rescale, freq_print_episode=freq_print_episode)
            del agent_params

    @staticmethod
    def train_agent(agent, env, config, agent_params, outdir, logger, reward_rescale, action_rescale, freq_print_episode=500):

        # TODO: test this
        xp = Trainer(agent          = agent,
                     env            = env,
                     env_config     = config,
                     agent_params   = agent_params,
                     logger         = logger,
                     reward_rescale = reward_rescale,
                     action_rescale = action_rescale)
        xp.train_agent(outdir, freq_print_episode)
        print('done\n\n')


