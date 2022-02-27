import numpy as np
from numpy.core.fromnumeric import argmax 
from Agents.Bandits.Bandits import UCB, LinUCB

if __name__ == '__main__':

    data = np.loadtxt('TP/TP1/CTR.txt', dtype=str)

    # Baselines

    random = 0
    cumulative = 0
    optimal = 0

    cumulative_rates = [0]*10
    T = [0]*10
    for i in range(data.shape[0]):
        clicks = [float(e) for e in data[i].split(":")[-1].split(';')]

        rd_index = np.random.randint(10)
        random += clicks[rd_index]

        mus = [c/t if t != 0 else 10**10 for c, t in zip(cumulative_rates, T)]
        cumulative += clicks[argmax(mus)]
        cumulative_rates[argmax(mus)] += clicks[argmax(mus)]
        T[argmax(mus)] += 1

        optimal += max(clicks)

    print('\n--- Average click rates per baseline strategy --- \n')
    print(f'Random pick:      {random/5000}')
    print(f'Cumulative rates: {cumulative/5000}')
    print(f'Optimal strategy: {optimal/5000}\n')

    # UCB

    print('\n--- Average click rate for UCB based policies ---\n')
    g = 0
    m = UCB(10)

    for i in range(data.shape[0]):
        rewards = np.array([float(e) for e in data[i].split(":")[2].split(';')])
        choosen = m.pick_arm(i+1)
        m.arms[choosen].update(rewards[choosen])
        g += rewards[choosen]

    print(f"UCB:    {g/5000}")

    # UCB-V

    print('\n--- Average click rate for UCB-V based policies ---\n')
    g = 0
    m = UCB(10, var=True)

    for i in range(data.shape[0]):
        rewards = np.array([float(e) for e in data[i].split(":")[2].split(';')])
        choosen = m.pick_arm(i + 1)
        m.arms[choosen].update(rewards[choosen])
        g += rewards[choosen]

    print(f"UCB-V:    {g / 5000}")



    # LinUCB

    print('\n--- Average click rate for LinUCB based policies ---\n')
    g = 0
    m = LinUCB(10, 5, 0.5)

    for i in range(data.shape[0]):
        context = np.array([float(e) for e in data[i].split(":")[1].split(';')])
        rewards = np.array([float(e) for e in data[i].split(":")[2].split(';')])
        choosen = m.pick_arm(context)
        m.arms[choosen].update(context, rewards[choosen])
        g += rewards[choosen]

    print(f"LinUCB: {g/5000}\n\n")
