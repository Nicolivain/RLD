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

from Core.Scheduler import TaskScheduler

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

    config_path = {'Pendulum'   : 'Config/env_config/config_random_pendulum.yaml',
                   'MountainCar': 'Config/env_config/config_random_mountain_car.yaml',
                   'LunarLander': 'Config/env_config/config_random_lunar.yaml',
                   'Gridworld'  : 'Config/env_config/config_random_gridworld.yaml',
                   'Cartpole'   : 'Config/env_config/config_random_cartpole.yaml',
                   'QLGridworld': 'Config/env_config/config_qleanring_gridworld.yaml',
                   }[env]

    agent_params = {
                    'layers'            : [256],
                    'batch_size'        : None,
                    'memory_size'       : 1000,
                    'k'                 : 5,
                    'use_dkl'           : True,
                    'reversed_dkl'      : False
                    }

    search_space = {
                    'gamma'                : [0.98, 0.99, 0.999],
                    'memory_size'          : [256, 512, 1024],
                    'learningRate'         : [0.01, 0.001, 0.0001],
                    }

    save_path = os.path.join(save_root_dir, env, mode)

    sch = TaskScheduler(models, search_space, config_path, agent_params, save_path, model_save_tag=mode)
    sch.search(n_training)
