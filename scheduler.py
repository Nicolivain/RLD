import os

from Agents.DQN.VanillaDQN import VanillaDQN
from Agents.DQN.DuelingDQN import DuelingDQN
from Agents.DQN.ReplayDQN import ReplayDQN
from Agents.DQN.TargetDQN import TargetDQN
from Agents.DQN.DoubleDQN import DoubleDQN
from Agents.DQN.DQN import DQN

from Agents.Policy.A2C import A2C
from Agents.Policy.PPO import AdaptativePPO
from Agents.Policy.ClippedPPO import ClippedPPO

from Agents.Continuous.SAC import SAC

from Agents.Continuous.DDPG import DDPG

from Agents.Imitation.BehavioralCloning import BehavioralCloning
from Agents.Imitation.GAIL import GAIL

from Core.Scheduler import TaskScheduler
from Tools.core import *


if __name__ == '__main__':

    # SEARCH CONFIG:

    mode  = {0: 'VanillaDQN',
             1: 'ReplayDQN',
             2: 'TargetDQN',
             3: 'DQN',
             4: 'DuelingDQN',
             5: 'DoubleDQN',
             6: 'A2C',
             7: 'PPO',
             8: 'ClippedPPO',
             9: 'DDPG',
             10: 'SAC',
             11: 'BC',
             12: 'GAIL'
             }[0]

    env   = {0: 'Pendulum',
             1: 'MontainCar',
             2: 'Gridworld',
             3: 'LunarLander',
             4: 'Cartpole',
             5: 'QLGridworld',
             6: 'LunarContinuous',
             7: 'GoalsGridworld'}[0]

    n_training = 20

    save_root_dir = 'XP'

    # SEARCH

    search_space = {
        'gamma'         : [0.98, 0.99, 0.999],
        'memory_size'   : [256, 512, 1024],
        'learning_rate' : [0.01, 0.001, 0.0001],
    }

    # You need to provide a search space matching the algorithm to tune, not all parameters are necessairly useful (see Config)

    models = {'VanillaDQN' : VanillaDQN,
              'ReplayDQN': ReplayDQN,
              'TargetDQN': TargetDQN,
              'DQN': DQN,
              'DuelingDQN': DuelingDQN,
              'DoubledQN': DoubleDQN,
              'A2C': A2C,
              'PPO': AdaptativePPO,
              'ClippedPPO': ClippedPPO,
              'DDPG': DDPG,
              'SAC': SAC,
              'BC': BehavioralCloning,
              'GAIL': GAIL
              }[mode]

    config_path = {'Pendulum'       : 'Config/env_config/config_random_pendulum.yaml',
                   'MountainCar'    : 'Config/env_config/config_random_mountain_car.yaml',
                   'LunarLander'    : 'Config/env_config/config_random_lunar.yaml',
                   'LunarContinuous': 'Config/env_config/config_random_lunar_countinuous.yaml',
                   'Gridworld'      : 'Config/env_config/config_random_gridworld.yaml',
                   'Cartpole'       : 'Config/env_config/config_random_cartpole.yaml',
                   'QLGridworld'    : 'Config/env_config/config_qleanring_gridworld.yaml',
                   }[env]

    agent_params = {
                    'VanillaDQN': 'Config/model_config/config_DQN.yaml',
                    'ReplayDQN' : 'Config/model_config/config_DQN.yaml',
                    'TargetDQN' : 'Config/model_config/config_DQN.yaml',
                    'DQN'       : 'Config/model_config/config_DQN.yaml',
                    'DuelingDQN': 'Config/model_config/config_DQN.yaml',
                    'DoubleDQN' : 'Config/model_config/config_DQN.yaml',
                    'A2C'       : 'Config/model_config/config_A2C.yaml',
                    'PPO'       : 'Config/model_config/config_PPO.yaml',
                    'ClippedPPO': 'Config/model_config/config_PPO.yaml',
                    'DDPG'      : 'Config/model_config/config_DDPG.yaml',
                    'SAC'       : 'Config/model_config/config_SAC.yaml',
                    'BC'        : 'Config/model_config/config_BC.yaml',
                    'GAIL'      : 'Config/model_config/config_GAIL.yaml'
                    }[mode]

    save_path = os.path.join(save_root_dir, env, mode)

    sch = TaskScheduler(models, search_space, config_path, agent_params, save_path, model_save_tag=mode)
    sch.search(n_training)
