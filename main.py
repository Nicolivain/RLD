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
             8: 'DDPG'}[4]

    env   = {0: 'Pendulum',
             1: 'MontainCar',
             2: 'Gridworld',
             3: 'LunarLander',
             4: 'Cartpole',
             5: 'QLGridworld'}[3]

    n_training = 20

    save_root_dir = 'XP'

    # SEARCH
    # Tous les params ne sont pas necessairement utiles pour tous les modeles

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
                   'Gridworld'  : 'Training/configs/config_random_gridworld.yaml',
                   'Cartpole'   : 'Training/configs/config_random_cartpole.yaml',
                   'QLGridworld': 'Training/configs/config_qleanring_gridworld.yaml',
                   }[env]

    agent_params = {'layers'            : [20, 20],
                    'batch_size'        : 64,
                    'memory_size'       : 1000,
                    'update_target'     : 100,
                    }

    search_space = {'batch_size'           : [64, 128, 256, 512],
                    'explo'                : [0.05, 0.1, 0.15, 0.2],
                    'gamma'                : [0.98, 0.99, 0.999],
                    'memory_size'          : [1000, 3000, 5000, 10000],
                    'learningRate'         : [0.01, 0.001, 0.0001],
                    'update_target'        : [100, 500, 1000]
                    }

    save_path = os.path.join(save_root_dir, env, mode)

    sch = TaskScheduler(models, search_space, config_path, agent_params, save_path, model_save_tag=mode)
    sch.search(n_training)
