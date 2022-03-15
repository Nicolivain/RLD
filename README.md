# RLD
## Reinforcement Learning and Advanced Deep Learning

### Course from Master M2A (DAC) @ Sorbonne Université Paris.
This course covers reinforcement learning algorithms and generative deep learning methods.  
Course website: https://dac.lip6.fr/master/rld-2021-2022/

### Getting started:
main.py contains an hyperparameters search tool. A main file for each algorithm/TME is available under 'TP' folder.
Hyperparameters of each algorithm can be tuned in 'Config/model_parameters', and then executed through the associated main function in 'TP'.

### Implemented Algorithms:
- UCB and LinUCB Bandits
- Policy and Value Iteration 
- QLearning, SARSA, DynaQ
- Deep Q Learning (minDQN), DuelingDQN, TargetDQN, Double VanillaDQN
- Goal VanillaDQN, Hindsight Experience Replay, Iterative Goal Sampling
- Actor Critic A2C
- Trusted Region Actor Critic PPO and Clipped PPO
- DDPG, Multi Agent DDPG
- SAC, Adaptative Temperature SAC
- Imitation Learning (GAIL)
- GAN, VAE
- Normalizing Flow: GLOW

### Environnement:
Grid World, Cartpole, Lunar Lander, Pendulum, MultiAgent
    
### Ressources:
- Lilian Weng's blog on Policy Algorithm: https://lilianweng.github.io/lil-log/2018/04/08/policy-gradient-algorithms.html 
