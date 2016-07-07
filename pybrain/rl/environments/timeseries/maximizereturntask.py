__author__ = 'Sam Stern, samuelostern@gmail.com'

#from .timeseries import AR1Environment

from pybrain.rl.environments.task import Task
from numpy import sign
from math import log

class MaximizeReturnTask(Task):

    def getReward(self):
        # TODO: make sure to check how to combine the returns (sum or product) depending on whether the timeseries is given in price, return or log returns
        # TODO: also, MUST check whether the reward is the cumulative reward or the one-step reward (it is currently the cumulative reward)
        # TODO: make sure that the returns array is always the first column of ts. EDIT
        t=self.env.time
        if t==10:
            pass
        delta=0.005
        latestAction=self.env.action[0]
        previousAction = self.env.actionHistory[-2]
        cost=delta*abs(sign(latestAction)-sign(previousAction))
        #ret=sum([log(1+self.env.ts[i]) for i in range(t,t+30)])
        ret=self.env.ts[t]
        #reward=ret*sign(latestAction)-cost
        #reward=ret*sign(latestAction)
        #reward=sum([log(1+(x/100))*sign(latestAction) for x in ret])#*sign(latestAction)
        reward=(1+(previousAction*ret))*(1-cost)-1
        self.env.incrementTime()


        return reward

    def reset(self):
        self


class NeuneierRewardTask(Task):
    """
    This task implements a reward structure as outlined in Neuneier(1996)
    """
    def getReward(self):
        # TODO: change the capital so that it reinvestes

        t=self.env.time
        c=1.0 #capital to invest at time t
        epsilon=0.1+(c/100) # transaction cost
        latestAction=self.env.action
        ret=self.env.ts[0,t-1]
        reward = (ret*latestAction)*(c-epsilon)-c
        return reward

