import matplotlib

from Agents.Policy.ClippedPPO import ClippedPPO
from Agents.Policy.PPO import AdaptativePPO
from Tools.core import *
from Tools.utils import *
from Core.Trainer import Trainer

matplotlib.use("Qt5agg")
# matplotlib.use("TkAgg")

if __name__ == '__main__':

    mode = ['PPO', 'ClippedPPO'][1]

    env, config, outdir, logger = init('Config/env_config/config_random_cartpole.yaml', mode, outdir=None,
                                       copy_config=False, launch_tb=False)
    params = load_model_params('PPO', env, config)
    agent = {'PPO': AdaptativePPO, 'ClippedPPO': ClippedPPO}[mode]

    xp = Trainer(agent          = agent,
                 env            = env,
                 env_config     = config,
                 agent_params   = params,
                 logger         = logger,
                 reward_rescale = 100,
                 action_rescale = 1)

    xp.train_agent(outdir)
