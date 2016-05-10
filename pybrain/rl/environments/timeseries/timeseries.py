__author__ = 'Sam Stern, samuelostern@gmail.com'

from random import gauss
from pybrain.rl.environments.environment import Environment
from numpy import array, matrix, append


class TSEnvironment(Environment):
    """ test time-series environment for artificial data-set"""
    def __init__(self):
        """Initialize environment randomly"""
        self.time=0
        self.action=[]
        self.actionHistory=[]
        self.ts=None
        #self.tsLength=tsLength
        #self.ts = TSEnvironment.importSnP()
        #self.ts = TSEnvironment.__createTS(tsLength, 0.9)

    def getSensors(self):
        """ the currently visible state of the world (the observation may be
            stochastic - repeated calls returning different values)

            :rtype: by default, this is assumed to be a numpy array of doubles
        """
        t=self.time
        return self.ts[0,t]

    def performAction(self, action):
        """ perform an action on the world that changes it's internal state (maybe
            stochastically).
            :key action: an action that should be executed in the Environment.
            :type action: by default, this is assumed to be a numpy array of doubles
        """
        self.action=action
        self.actionHistory.append(action)
        self.time+=1

    def reset(self):
        """set time back to the start"""
        self.time=0

    @property
    def indim(self):
        return len(self.action)

    @property
    def outdim(self):
        return len(self.sensors)


# Special case of AR(1) process

class AR1Environment(TSEnvironment):

    def __init__(self,tsLength):
        super(AR1Environment,self).__init__()
        self.tsLength=tsLength
        self.rho=0.99 #order of autoregression
        self.ts=AR1Environment.__createTS(tsLength,self.rho)

    @staticmethod
    def __createTS(tsLength,rho):
        ts = matrix([0.0 for x in range(tsLength)])
        for i in range(1,tsLength):
            ts.put([0,i],rho*ts[0,i-1]+gauss(0,0.2))
        return ts

# Special case of SnP returns Environment

class SnPEnvironment(TSEnvironment):
    def __init__(self):
        super(SnPEnvironment,self).__init__()
        self.ts=SnPEnvironment.__importSnP()

    @staticmethod
    def __importSnP():
        import csv
        with open('SnP_data.csv','r') as f:
            data = [row for row in csv.reader(f.read().splitlines())]
        price=[]
        [price.append(data[i][4]) for i in range(1,len(data))]
        price.reverse()
        price=map(float,price)
        rets=matrix([(price[i]-price[i-1])/price[i-1] for i in range(len(price))])
        return rets