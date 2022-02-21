import matplotlib

matplotlib.use("TkAgg")
import numpy as np
from Agents.Continuous.MADDPG import *
from Tools.utils import *
from Core.Trainer import *


if __name__ == '__main__':

    #CONSEIL POUR DEBUG : copier le fichier simple_spread.py pour mettre le nombre de cible à 1 et avec 1 un seul agent

    # map = ['simple_spread','simple_adversary', 'simple_tag'][0]
    #
    # env,scenario,world = make_env(map)
    # config = load_yaml('Config/env_config/config_maddpg_simple_spread.yaml')

    env, _, world, config, outdir, logger = init_MADDPG('Config/env_config/config_maddpg_simple_spread.yaml', 'MADDPG', outdir=None,
                                       copy_config=False, launch_tb=False)

    # #Handmade Features extractor
    # junk = env.reset()
    # nb_agents = len(env.agents)  # len(junk) #nombre d'agents
    # dim = len(junk[0])  # dimension de l'espace d'observations
    # print(world)
    # print("Nb d'agents:", nb_agents, "\n Dim_space :", dim, "\n Dim_action :", world.dim_p)

    params = load_model_params('MADDPG', env, config, world)
    agent = MADDPG
    xp = Trainer_MADDPG(agent=agent,
                 env=env,
                 env_config=config,
                 agent_params=params,
                 logger=logger,
                 reward_rescale=1,
                 action_rescale=1.0)

    xp.train_agent(outdir)