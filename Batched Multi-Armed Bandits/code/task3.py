"""
NOTE: You are only allowed to edit this file between the lines that say:
    # START EDITING HERE
    # END EDITING HERE

This file contains the AlgorithmManyArms class. Here are the method details:
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
"""

import numpy as np

# START EDITING HERE
# You can use this space to define any helper functions that you need
# END EDITING HERE


class AlgorithmManyArms:
    def __init__(self, num_arms, horizon):
        # Horizon is same as number of arms
        # START EDITING HERE
        self.num_arms = num_arms
        self.t=1
        self.values=np.zeros(num_arms)
        self.counts=np.zeros(num_arms)
        # You can add any other variables you need here
        # END EDITING HERE
    def softmax(self):
        lst=[]
        for i in range(self.num_arms):
            if not (np.abs(self.values[i])<=1e-12):
                lst.append(i)
        arr=self.values[lst].copy()
        sft_max=np.exp(arr)
        sm=sum(sft_max)
        sft_max=sft_max/sm
        ret=self.values.copy()
        ret[lst]=sft_max
        if not (np.abs(sum(ret)-1)<=1e-12):
            print('why?')
        # print(ret.shape,self.num_arms)
        return ret   

    def give_pull(self):
        # START EDITING HERE
        rnd=np.random.random()
        thr=min(1,np.log(self.num_arms)/self.t)
        if(rnd<=thr):
            return np.random.randint(self.num_arms)
        else:
            # return np.random.choice(self.num_arms,p=((self.values+1e-12)/sum(self.values+1e-12)))
            return np.argmax(self.values)

        # END EDITING HERE
    
    def get_reward(self, arm_index, reward):
        # START EDITING HERE
        self.t+=1
        self.counts[arm_index] += 1
        n = self.counts[arm_index]
        value = self.values[arm_index]
        new_value = ((n - 1) / n) * value + (1 / n) * reward
        self.values[arm_index] = new_value
        pass
        # END EDITING HERE
