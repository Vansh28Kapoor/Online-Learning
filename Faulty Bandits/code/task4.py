"""
NOTE: You are only allowed to edit this file between the lines that say:
    # START EDITING HERE
    # END EDITING HERE

This file contains the MultiBanditsAlgo class. Here are the method details:
    - __init__(self, num_arms, horizon): This method is called when the class
        is instantiated. Here, you can add any other member variables that you
        need in your algorithm.
    
    - give_pull(self): This method is called when the algorithm needs to
        select an arm to pull. The method should return the index of the arm
        that it wants to pull (0-indexed).
    
    - get_reward(self, arm_index, set_pulled, reward): This method is called 
        just after the give_pull method. The method should update the 
        algorithm's internal state based on the arm that was pulled and the 
        reward that was received.
        (The value of arm_index is the same as the one returned by give_pull 
        but set_pulled is the set that is randomly chosen when the pull is 
        requested from the bandit instance.)
"""

import numpy as np

# START EDITING HERE
# You can use this space to define any helper functions that you need
# END EDITING HERE


# class MultiBanditsAlgo:
#     def __init__(self, num_arms, horizon):
#         # You can add any other variables you need here
#         self.num_arms = num_arms
#         self.horizon = horizon
#         self.counts=np.zeros((2,num_arms))
#         self.values=np.zeros((2,num_arms))
#         self.means=np.zeros(num_arms)
        # START EDITING HERE
        # self.sucess=np.zeros((2,num_arms))
        # self.fail=np.zeros((2,num_arms))
        # self.sampling=np.zeros(num_arms,float)
        # self.t=0

        
    #     # END EDITING HERE
    # def klucb(self,start,index,c=3,sens=1e-12,end=1.0):
    #     mean=(start+end)/2.0
    #     dest=(np.log(self.t)+ c*(np.log(np.log(self.t)))/self.counts[index])
    #     sign=kl(self.values[index],mean)-dest
    #     if(end-start<=sens or np.abs(sign)<1e-5):
    #         return mean
    #     else:
    #         if (sign>0):
    #             return self.klucb(start=start,end=mean,index=index,c=c,sens=sens)
    #         else:
    #             return self.klucb(start=mean,end=end,index=index,c=c,sens=sens)
    # def give_pull(self):
    #     if 0 in self.counts:
    #         return np.argwhere(0==self.counts)[0]
    #     else:
    #         return np.argmax(self.kl_ucb)

    # def get_reward(self, arm_index, set_pulled, reward):
    #     self.t +=1
    #     self.counts[set_pulled,arm_index]+=1
    #     n=self.counts[set_pulled,arm_index]
    #     value=self.values[set_pulled,arm_index]
    #     self.values[set_pulled,arm_index]=((n - 1) / n) * value + (1 / n) * reward
    #     if 0 not in (self.counts[0]+self.counts[1]):
    #         self.mean=(self.values[0]*self.counts[0]+self.values[1]*self.counts[1])/(self.counts[1]+self.counts[0])
    #     if 0 not in self.counts:
    #         for i in range(self.num_arms):
    #             max=max()
    #             diff=kl()
    #             self.kl_ucb[i]=self.klucb(start=self.values[i],index=i,c=2,sens=1e-12,end=1.0)


class MultiBanditsAlgo:
    def __init__(self, num_arms, horizon):
        # You can add any other variables you need here
        self.num_arms = num_arms
        self.horizon = horizon
        self.sucess=np.zeros((2,num_arms))
        self.fail=np.zeros((2,num_arms))
        self.sampling=np.zeros(num_arms,float)
    
    def give_pull(self):
        # START EDITING HERE
        alpha=self.sucess+1
        beta=self.fail +1
        sampling_2=self.sampling.copy()
        lst=np.random.binomial(1, 0.5, self.num_arms)
        for i in range(self.num_arms):
            sampling_2[i]=np.random.beta(alpha[0,i]+alpha[1,i],beta[0,i]+beta[1,i])
        self.sampling=sampling_2
        return np.argmax(self.sampling)
        

        # END EDITING HERE
    
    def get_reward(self, arm_index, set_pulled, reward):
        # START EDITING HERE
        if(abs(reward)<1e-12):
            self.fail[set_pulled,arm_index]+=1
        else:
            self.sucess[set_pulled,arm_index]+=1
        pass
        # END EDITING HERE

