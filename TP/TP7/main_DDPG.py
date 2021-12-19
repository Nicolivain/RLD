import matplotlib

from Agents.Continuous.DDPG import DDPG
from Tools.core import *
from Tools.utils import *
from Core.Trainer import Trainer

matplotlib.use("Qt5agg")
# matplotlib.use("TkAgg")

if __name__ == '__main__':
    env, config, outdir, logger = init('Config/env_config/config_random_pendulum.yaml', 'DDPG', outdir=None,
                                       copy_config=False, launch_tb=False)
    params = load_model_params('DDPG', env, config)
    agent = DDPG

    xp = Trainer(agent=agent,
                 env=env,
                 env_config=config,
                 agent_params=params,
                 logger=logger,
                 reward_rescale=10,
                 action_rescale=2.0)

    xp.train_agent(outdir)
