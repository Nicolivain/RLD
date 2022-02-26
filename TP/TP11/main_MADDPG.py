import matplotlib

matplotlib.use("TkAgg")
import numpy as np
from Agents.Continuous.MADDPG import *
from Tools.utils import *
from Core.Trainer import *


if __name__ == '__main__':

    #CONSEIL POUR DEBUG : copier le fichier simple_spread.py pour mettre le nombre de cible à 1 et avec 1 un seul agent

    # Choose your map env in the config file

    env, _, world, config, outdir, logger = init_MADDPG('Config/env_config/config_multiagent.yaml', 'MADDPG', outdir=None,
                                       copy_config=False, launch_tb=False)

    params = load_model_params('MADDPG', env, config, world)
    print(params)
    agent = MADDPG
    xp = Trainer_MADDPG(agent=agent,
                 env=env,
                 env_config=config,
                 agent_params=params,
                 logger=logger,
                 reward_rescale=1, #between 0 and 1
                 action_rescale=1) #between 0 and 1

    xp.train_agent(outdir)