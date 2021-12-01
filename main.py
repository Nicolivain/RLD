import os

from Agents.Continuous.DDPG import DDPG
from Tools.scheduler import TaskScheduler
from Tools.core import *


if __name__ == '__main__':

    # SEARCH CONFIG:

    mode  = ['DDPG'][0]
    env   = ['pendulum'][0]

    n_training = 5

    save_root_dir = 'XP'


    # SEARCH
    # Tous les params ne sont pas nécéssairement utiles pour tous les modèles

    models = {'DDPG': DDPG,
              }[mode]

    config_path = {'pendulum': 'Training/configs/config_random_pendulum.yaml',
                   }[env]

    agent_params = {'layers': [20, 20],
                    'batch_per_learn': 3,
                    'memory_size': 10000,
                    'batch_size': 1000,
                    'rho': 0.99
                    }

    search_space = {'batch_per_learn'      : [1, 3, 5, 10],
                    'batch_size'           : [64, 128, 256, 512, 1024, 2048],
                    'rho'                  : [0.9, 0.95, 0.99, 0.999],
                    'freqOptim'            : [100, 1000, 2000, 5000, 10000],
                    'explo'                : [0.01, 0.05, 0.1, 0.15, 0.2],
                    'gamma'                : [0.9, 0.95, 0.99, 0.999, 0.9999],
                    'learningRate'         : [0.01, 0.001, 0.0005, 0.0001],
                    'p_learningRate'       : [0.01, 0.001, 0.0005, 0.0001],
                    'q_learningRate'       : [0.01, 0.001, 0.0005, 0.0001],
                    }

    save_path = os.path.join(save_root_dir, mode)

    sch = TaskScheduler(models, search_space, config_path, agent_params, save_path, model_save_tag=mode)
    sch.search(n_training)
