import numpy as np
from numpy.core.fromnumeric import argmax
from numpy.linalg import inv

class UCBArm():
    def __init__(self,index):
        self.index = index
        self.T = 0
        self.rewards = 0

    def compute_b(self, step, var):
        if var == True :
            v = np.var(self.rewards) #empirical variance
            b = self.rewards/self.T + np.sqrt(2*np.log(step)*v/self.T) + np.log(step)/(2*self.T) if self.T != 0 else 10**10
        else :
            b = self.rewards/self.T + np.sqrt(2*np.log(step)/self.T) if self.T !=0 else 10**10
        return b

    def update(self, reward):
        self.T += 1
        self.rewards += reward

class UCB():
    def __init__(self, n_arms, var=False):
        self.n_arms = n_arms
        self.arms = [UCBArm(i) for i in range(n_arms)]
        self.var = var

    def pick_arm(self, step):
        bs = [a.compute_b(step, self.var) for a in self.arms]
        return argmax(bs)

class LinUCBArm():
    def __init__(self, index, d, alpha):
        self.index=index
        self.A = np.eye(d)
        self.alpha = alpha
        self.b = np.zeros(d)
    
    def compute_p(self, x):
        invA = inv(self.A)
        theta_hat = invA@self.b
        p = theta_hat.T@x + self.alpha * np.sqrt(x.T@invA@x)
        return p 

    def update(self, x, reward):
        self.A += np.outer(x.T, x)
        self.b += reward * x

class LinUCB():
    def __init__(self, n_arms, d, alpha):
        self.n_arms = n_arms
        self.d = d 
        self.alpha = alpha
        self.arms = [LinUCBArm(i, self.d, self.alpha) for i in range(self.n_arms)]

    def pick_arm(self, context):
        ps = [a.compute_p(context) for a in self.arms]
        return argmax(ps)
        


