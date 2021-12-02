import os

from Agents.DQN.DQN import DQN
from Agents.DQN.DuelingDQN import DuelingDQN
from Agents.DQN.ReplayDQN import ReplayDQN
from Agents.DQN.TargetDQN import TargetDQN
from Agents.DQN.minDQN import MinDQN

from Agents.Policy.A2C import A2C
from Agents.Policy.PPO import AdaptativePPO
from Agents.Policy.ClippedPPO import ClippedPPO

from Agents.Continuous.DDPG import DDPG

from Tools.scheduler import TaskScheduler
from Tools.core import *


if __name__ == '__main__':

    # SEARCH CONFIG:

    mode  = {0: 'DQN',
             1: 'ReplayDQN',
             2: 'TargetDQN',
             3: 'minDQN',
             4: 'DuelingDQN',
             5: 'A2C',
             6: 'PPO',
             7: 'ClippedPPO',
             8: 'DDPG'}[8]

    env   = ['Pendulum', 'MontainCar', 'Gridworld', 'LunarLander', 'Cartpole', 'QLGridworld'][0]

    n_training = 50

    save_root_dir = 'XP'

    # SEARCH
    # Tous les params ne sont pas nécéssairement utiles pour tous les modèles

    models = {'DQN' : DQN,
              'ReplayDQN': ReplayDQN,
              'TargetDQN': TargetDQN,
              'minDQN': MinDQN,
              'DuelingDQN': DuelingDQN,
              'A2C': A2C,
              'PPO': AdaptativePPO,
              'ClippedPPO': ClippedPPO,
              'DDPG': DDPG,
              }[mode]

    config_path = {'Pendulum'   : 'Training/configs/config_random_pendulum.yaml',
                   'MountainCar': 'Training/configs/config_random_mountain_car.yaml',
                   'LunarLander': 'Training/configs/config_random_lunar.yaml',
                   'Gridword'   : 'Training/configs/config_random_gridworld.yaml',
                   'Cartpole'   : 'Training/configs/config_random_cartpole.yaml',
                   'QLGridworld': 'Training/configs/config_qleanring_gridworld.yaml',
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

    save_path = os.path.join(save_root_dir, env, mode)

    sch = TaskScheduler(models, search_space, config_path, agent_params, save_path, model_save_tag=mode)
    sch.search(n_training)
