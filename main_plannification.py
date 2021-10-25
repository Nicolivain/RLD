import matplotlib
import gym
import Tools.gridworld
from Agents.Plannification.PolicyIteration import PolicyIterationAgent
from Agents.Plannification.ValueIteration import ValueIterationAgent
import numpy as np


matplotlib.use("TkAgg")

if __name__ == '__main__':

    env = gym.make("gridworld-v0")
    env.setPlan("gridworldPlans/plan0.txt", {0: -0.001, 3: 1, 4: 1, 5: -1, 6: -1})

    env.seed(0)  # Initialise le seed du pseudo-random
    np.random.seed(0)

    env.render()  # permet de visualiser la grille du jeu
    env.render(mode="human")  # visualisation sur la console
    states, mdp = env.getMDP()  # récupère le mdp et la liste d'états
    print("Nombre d'etats : ", len(states))
    state, transitions = list(mdp.items())[0]

    # Execution avec un Agent
    epsilon = 1e-10
    gamma = 0.99

    agent = [ValueIterationAgent(env.action_space, env),
             PolicyIterationAgent(env.action_space, env)][1]

    agent.compute_policy(epsilon, gamma)

    episode_count = 100
    reward = 0
    done = False
    rsum = 0

    for i in range(episode_count):
        obs = env.reset()
        env.verbose = (i % 100 == 0 and i > 0)  # afficher 1 episode sur 100
        if env.verbose:
            env.render()
        j = 0
        rsum = 0
        while True:
            action = agent.act(obs)
            obs, reward, done, _ = env.step(action)
            rsum += reward
            j += 1
            if env.verbose:
                env.render()
            if done:
                print("Episode : " + str(i) + " rsum=" + str(rsum) + ", " + str(j) + " actions")
                break

    print("done")
    env.close()
