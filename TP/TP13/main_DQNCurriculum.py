import matplotlib
import torch

from Agents.DQN.DQNGoal import DQNGoal
from Agents.DQN.IGS import IGS
from Tools.core import *
from Tools.utils import *
from Structure.Memory import Memory
from tqdm import tqdm
import Tools.gridworld
from Core.Trainer import TrainerDQNGoal, TrainerHER, TrainerIGS

# matplotlib.use("Qt5agg")
matplotlib.use("TkAgg")

if __name__ == '__main__':

    mode = ['DQNGoal', 'HER', 'IGS'][0]

    # For IGS, take Plan3.txt map in config file
    # For HER, take Plan2.txt map in config file
    # For GoalDQN, take Plan2Multi.txt in config file

    env, config, outdir, logger = init('Config/env_config/config_goals_gridworld.yaml', mode, outdir=None)
    params = load_model_params('DQNCurriculum', env, config)
    print(params)

    agent = {'DQNGoal': DQNGoal, 'HER': DQNGoal, 'IGS': IGS}[mode]
    Trainer = {'DQNGoal': TrainerDQNGoal, 'HER': TrainerHER, 'IGS': TrainerIGS}[mode]

    xp = Trainer(agent          = agent,
                 env            = env,
                 env_config     = config,
                 agent_params   = params,
                 logger         = logger,
                 reward_rescale = 1,
                 action_rescale = 1)

    xp.train_agent(outdir)
