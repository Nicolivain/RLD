import matplotlib

matplotlib.use("TkAgg")

"""
Code for creating a multiagent environment with one of the scenarios listed
in ./scenarios/.
Can be called by using, for example:
    env = make_env('simple_speaker_listener')
After producing the env object, can be used similarly to an OpenAI gym
environment.

A policy using this environment must output actions in the form of a list
for all agents. Each element of the list should be a numpy array,
of size (env.world.dim_p + env.world.dim_c, 1). Physical actions precede
communication actions in this array. See environment.py for more details.
"""

def make_env(scenario_name, benchmark=False):
    '''
    Creates a MultiAgentEnv object as env. This can be used similar to a gym
    environment by calling env.reset() and env.step().
    Use env.render() to view the environment on the screen.

    Input:
        scenario_name   :   name of the scenario from ./scenarios/ to be Returns
                            (without the .py extension)
        benchmark       :   whether you want to produce benchmarking data
                            (usually only done during evaluation)

    Some useful env properties (see environment.py):
        .observation_space  :   Returns the observation space for each agent
        .action_space       :   Returns the action space for each agent
        .n                  :   Returns the number of Agents
    '''
    from Env_MultiAgent.multiagent import MultiAgentEnv
    from Env_MultiAgent import multiagent as scenarios

    # load scenario from script
    scenario = scenarios.load(scenario_name + ".py").Scenario()
    # create world
    world = scenario.make_world()
    # create multiagent environment
    world.dim_c = 0
    if benchmark:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, scenario.benchmark_data)
    else:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)
    env.discrete_action_space = False
    env.discrete_action_input = False
    scenario.reset_world(world)
    return env,scenario,world

if __name__ == '__main__':

    #CONSEIL POUR DEBUG : copier le fichier simple_spread.py pour mettre le nombre de cible à 1 et avec 1 un seul agent

    rule = ['simple_spread','simple_adversary', 'simple_tag'][2]

    env,scenario,world = make_env(rule)
    junk = env.reset()
    nb_agents = len(env.agents)#len(junk) #nombre d'agents
    dim = len(junk[0]) #dimension de l'espace d'observations
    print(nb_agents,dim, world.dim_p)

    reward = []
    for _ in range(100):
        break
        a = []
        for i, _ in enumerate(env.agents):
            a.append((np.random.rand(2)-0.5)*2)
        o, r, d, i = env.step(a.copy()) # /!\ attention de bien mettre une copie en argument, le step modifie l'action !!
        print(o, r, d, i)

        reward.append(r)
        env.render(mode="none")
    print(reward)


    env.close()