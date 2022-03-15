import os

from Agents.DQN.VanillaDQN import VanillaDQN
from Agents.DQN.DuelingDQN import DuelingDQN
from Agents.DQN.ReplayDQN import ReplayDQN
from Agents.DQN.TargetDQN import TargetDQN
from Agents.DQN.DQN import DQN

from Agents.Policy.A2C import A2C
from Agents.Policy.PPO import AdaptativePPO
from Agents.Policy.ClippedPPO import ClippedPPO

from Agents.Continuous.DDPG import DDPG

from Core.Scheduler import TaskScheduler
from Tools.core import *


if __name__ == '__main__':

    # SEARCH CONFIG:

    mode  = {0: 'VanillaDQN',
             1: 'ReplayDQN',
             2: 'TargetDQN',
             3: 'minDQN',
             4: 'DuelingDQN',
             5: 'A2C',
             6: 'PPO',
             7: 'ClippedPPO',
             8: 'DDPG'}[6]

    env   = {0: 'Pendulum',
             1: 'MontainCar',
             2: 'Gridworld',
             3: 'LunarLander',
             4: 'Cartpole',
             5: 'QLGridworld'}[3]

    n_training = 20

    save_root_dir = 'XP'

    # SEARCH

    search_space = {
        'gamma'         : [0.98, 0.99, 0.999],
        'memory_size'   : [256, 512, 1024],
        'learning_rate' : [0.01, 0.001, 0.0001],
    }

    # Tous les params ne sont pas necessairement utiles pour tous les modeles

    models = {'VanillaDQN' : VanillaDQN,
              'ReplayDQN': ReplayDQN,
              'TargetDQN': TargetDQN,
              'minDQN': DQN,
              'DuelingDQN': DuelingDQN,
              'A2C': A2C,
              'PPO': AdaptativePPO,
              'ClippedPPO': ClippedPPO,
              'DDPG': DDPG,
              }[mode]

    config_path = {'Pendulum'   : 'Config/env_config/config_random_pendulum.yaml',
                   'MountainCar': 'Config/env_config/config_random_mountain_car.yaml',
                   'LunarLander': 'Config/env_config/config_random_lunar.yaml',
                   'Gridworld'  : 'Config/env_config/config_random_gridworld.yaml',
                   'Cartpole'   : 'Config/env_config/config_random_cartpole.yaml',
                   'QLGridworld': 'Config/env_config/config_qleanring_gridworld.yaml',
                   }[env]

    agent_params = {
                    'VanillaDQN'       : 'Config/model_config/config_DQN.yaml',
                    'ReplayDQN' : 'Config/model_config/config_DQN.yaml',
                    'TargetDQN' : 'Config/model_config/config_DQN.yaml',
                    'minDQN'    : 'Config/model_config/config_DQN.yaml',
                    'Dueling'   : 'Config/model_config/config_DQN.yaml',
                    'A2C'       : 'Config/model_config/config_A2C.yaml',
                    'PPO'       : 'Config/model_config/config_PPO.yaml',
                    'ClippedPPO': 'Config/model_config/config_PPO.yaml',
                    'DDPG'      : 'Config/model_config/config_DDPG.yaml',
                    'SAC'       : 'Config/model_config/config_SAC.yaml',
                    }[mode]

    save_path = os.path.join(save_root_dir, env, mode)

    sch = TaskScheduler(models, search_space, config_path, agent_params, save_path, model_save_tag=mode)
    sch.search(n_training)
