import matplotlib

from Agents.DQN.DQN import DQN
from Agents.DQN.DuelingDQN import DuelingDQN
from Agents.DQN.ReplayDQN import ReplayDQN
from Agents.DQN.TargetDQN import TargetDQN
from Agents.DQN.minDQN import MinDQN
from Tools.core import *
from Tools.utils import *
import Tools.gridworld
from Core.Trainer import Trainer

# matplotlib.use("Qt5agg")
matplotlib.use("TkAgg")

if __name__ == '__main__':

    mode = ['DQN', 'ReplayDQN', 'TargetDQN', 'minDQN', 'DuelingDQN'][0]

    env, config, outdir, logger = init('Config/env_config/config_random_cartpole.yaml', mode, outdir=None)
    params = load_model_params('DQN', env, config)
    agent = {'DQN': DQN, 'ReplayDQN': ReplayDQN, 'TargetDQN': TargetDQN, 'minDQN': MinDQN, 'DuelingDQN': DuelingDQN}[mode]

    xp = Trainer(agent=agent,
                 env=env,
                 env_config=config,
                 agent_params=params,
                 logger=logger,
                 reward_rescale=100,
                 action_rescale=1)

    xp.train_agent(outdir)
