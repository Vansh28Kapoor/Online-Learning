"""
NOTE: You are only allowed to edit this file between the lines that say:
    # START EDITING HERE
    # END EDITING HERE

This file contains the base Algorithm class that all algorithms should inherit
from. Here are the method details:
    - __init__(self, num_arms, horizon): This method is called when the class
        is instantiated. Here, you can add any other member variables that you
        need in your algorithm.
    
    - give_pull(self): This method is called when the algorithm needs to
        select an arm to pull. The method should return the index of the arm
        that it wants to pull (0-indexed).
    
    - get_reward(self, arm_index, reward): This method is called just after the 
        give_pull method. The method should update the algorithm's internal
        state based on the arm that was pulled and the reward that was received.
        (The value of arm_index is the same as the one returned by give_pull.)

We have implemented the epsilon-greedy algorithm for you. You can use it as a
reference for implementing your own algorithms.
"""

import numpy as np
import math
# Hint: math.log is much faster than np.log for scalars

class Algorithm:
    def __init__(self, num_arms, horizon):
        self.num_arms = num_arms
        self.horizon = horizon
    
    def give_pull(self):
        raise NotImplementedError
    
    def get_reward(self, arm_index, reward):
        raise NotImplementedError

# Example implementation of Epsilon Greedy algorithm
class Eps_Greedy(Algorithm):
    def __init__(self, num_arms, horizon):
        super().__init__(num_arms, horizon)
        # Extra member variables to keep track of the state
        self.eps = 0.1
        self.counts = np.zeros(num_arms)
        self.values = np.zeros(num_arms)
    
    def give_pull(self):
        if np.random.random() < self.eps:
            return np.random.randint(self.num_arms)
        else:
            return np.argmax(self.values)
    
    def get_reward(self, arm_index, reward):
        self.counts[arm_index] += 1
        n = self.counts[arm_index]
        value = self.values[arm_index]
        new_value = ((n - 1) / n) * value + (1 / n) * reward
        self.values[arm_index] = new_value

# START EDITING HERE
def kl(a,b):
    if (np.abs(a)<=1e-12):
        s=-np.log(1-b)
    elif(np.abs(a-1)<=1e-12):
        s=-np.log(b)
    else:
        s=a*np.log(a/b)+(1-a)*np.log((1-a)/(1-b))
    return s 


# You can use this space to define any helper functions that you need
# END EDITING HERE

class UCB(Algorithm):
    def __init__(self, num_arms, horizon):
        super().__init__(num_arms, horizon)
        # START EDITING HERE
        self.values=np.zeros(num_arms,float)
        self.ucb=np.zeros(num_arms,float)
        self.counts=np.zeros(num_arms,int)
        self.t=0
        # END EDITING HERE
    
    def give_pull(self):
        # START EDITING HERE
        self.t +=1
        if 0 in self.counts:
            return np.where(0==self.counts)[0][0]
        else:
            return np.argmax(self.ucb)
        # END EDITING HERE  
        
    
    def get_reward(self, arm_index, reward):
        # START EDITING HERE
        self.counts[arm_index]+=1
        n=self.counts[arm_index].astype(float)
        value=self.values[arm_index]
        self.values[arm_index]=((n - 1) / n) * value + (1 / n) * reward
        for i in range(int(self.num_arms)):
            if (self.counts[i]!=0):
                self.ucb[i]=self.values[i]+np.sqrt(2*np.log(self.t)/self.counts[i])
        pass
        # END EDITING HERE


class KL_UCB(Algorithm):
    def __init__(self, num_arms, horizon):
        super().__init__(num_arms, horizon)
        # You can add any other variables you need here
        # START EDITING HERE
        self.values=np.zeros(self.num_arms,float)
        self.kl_ucb=np.zeros(num_arms,float)
        self.counts=np.zeros(num_arms,int)
        self.t=0 
        # END EDITING HERE
    def klucb(self,start,index,c=0,sens=1e-12,end=1.0):
        mean=(start+end)/2.0
        dest=(np.log(self.t)+ c*np.log(np.log(self.t)))/self.counts[index]
        sign=kl(self.values[index],mean)-dest
        if(end-start<=sens or np.abs(sign)<1e-5):
            return mean
        else:
            if (sign>0):
                return self.klucb(start=start,end=mean,index=index,c=c,sens=sens)
            else:
                return self.klucb(start=mean,end=end,index=index,c=c,sens=sens)
    
    def give_pull(self):
        # START EDITING HERE
        if 0 in self.counts:
            return np.where(0==self.counts)[0][0]
        else:
            return np.argmax(self.kl_ucb) 
        # END EDITING HERE
    
    def get_reward(self, arm_index, reward):
        # START EDITING HERE
        self.t +=1
        self.counts[arm_index]+=1
        n=self.counts[arm_index]
        value=self.values[arm_index]
        self.values[arm_index]=((n - 1) / n) * value + (1 / n) * reward
        if 0 not in self.counts:
            for i in range(self.num_arms):
                self.kl_ucb[i]=self.klucb(start=self.values[i],index=i,c=0,sens=1e-6,end=1.0)
        # END EDITING HERE

class Thompson_Sampling(Algorithm):
    def __init__(self, num_arms, horizon):
        super().__init__(num_arms, horizon)
        # You can add any other variables you need here
        # START EDITING HERE
        self.values=np.zeros(self.num_arms,float)
        self.counts=np.zeros(num_arms,int)
        self.sampling=np.zeros(num_arms,float)
        # END EDITING HERE
    
    def give_pull(self):
        # START EDITING HERE
        alpha=self.counts*self.values+1
        beta=(1-self.values)*self.counts+1
        for i in range(self.num_arms):
            self.sampling[i]=np.random.beta(alpha[i],beta[i])
        return np.argmax(self.sampling)
        # END EDITING HERE
    
    def get_reward(self, arm_index, reward):
        # START EDITING HERE
        self.counts[arm_index] += 1
        n = self.counts[arm_index]/1.0
        value = self.values[arm_index]
        new_value = ((n - 1) / n) * value + (1 / n) * reward
        self.values[arm_index] = new_value
        pass
        # END EDITING HERE
