import matplotlib
import numpy
from Agents.Continuous.SAC import SAC
from Tools.core import *
from Tools.utils import *

from Core.Trainer import Trainer

matplotlib.use("Qt5agg")
# matplotlib.use("TkAgg")

if __name__ == '__main__':

    torch.manual_seed(1)
    numpy.random.seed(1)

    env, config, outdir, logger = init('Config/env_config/config_random_lunar_continuous.yaml', 'SAC', outdir=None, copy_config=False, launch_tb=False)
    params = load_model_params('SAC', env, config)
    agent = SAC

    xp = Trainer(agent          = agent,
                 env            = env,
                 env_config     = config,
                 agent_params   = params,
                 logger         = logger,
                 reward_rescale = 100,
                 action_rescale = 1.0)  # output of env.action_space.high[0].item()

    xp.train_agent(outdir)
