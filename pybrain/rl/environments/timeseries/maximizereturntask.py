__author__ = 'Sam Stern, samuelostern@gmail.com'

#from .timeseries import AR1Environment

from pybrain.rl.environments.task import Task
from numpy import sign

class MaximizeReturnTask(Task):

    def getReward(self):
        # TODO: make sure to check how to combine the returns (sum or product) depending on whether the timeseries is given in price, return or log returns
        # TODO: also, MUST check whether the reward is the cumulative reward or the one-step reward (it is currently the cumulative reward)
        # TODO: make sure that the returns array is always the first column of ts
        t=self.env.time
        latestAction=self.env.action[0]
        previousAction = self.env.actionHistory[t-1]
        ret=self.env.ts[t,0]
        reward=ret*sign(latestAction)

        #actionHist=self.env.actionHistory
        #rets=self.env.ts[0,0:t+1].tolist()[0]
        #fun = lambda x,y :x*y
        #retsMade=map(fun,rets,actionHist)
        #reward=sum(retsMade)
        return reward

    def reset(self):
        self

class DifferentialSharpeRatioTask(Task):

    """
    Differential Sharpe Ratio as presented in J. Moody and L. Wu (1997). The reward is the first order term
    in the expansion of a moving average Sharpe ratio. The first order(/derivative) is used as it gives the marginal
    utility of taking an action
    """

    def __init__(self,env):
        super(DifferentialSharpeRatioTask,self).__init__(env)
        self.A=0#self.env.ts[0,0]
        self.B=0.01#self.env.ts[0,0]**2

    def getReward(self):
        a_t_Minus1=self.A
        b_t_Minus1=self.B

        t=self.env.time
        lastAction=self.env.action
        ts_ret=self.env.ts[0,t]
        r_t=ts_ret*lastAction
        #actionHist=self.env.actionHistory
        #dailyRets=self.env.ts[0,0:t-1].tolist()[0]
        #fun = lambda x,y :x*y
        #retsMade=map(fun,dailyRets,actionHist)
        #r_t=sum(retsMade)

        delA=r_t-a_t_Minus1
        delB=(r_t**2)-b_t_Minus1

        numeratorD=(b_t_Minus1*delA)-0.5*(a_t_Minus1*delB)
        denominatorD=(b_t_Minus1-(a_t_Minus1**2))**(3./2)
        d=numeratorD/denominatorD

        #update moments
        epsilon=0.01 # smaller epsilon makes it less inclined to trade
        self.A+=epsilon*delA
        self.B+=epsilon*delB

        return d

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

