__author__ = 'Sam Stern, samuelostern@gmail.com'

from random import gauss
from pybrain.rl.environments.environment import Environment
from numpy import array, matrix, empty, append
from math import log
import pandas as pd
import csv
import time


class TSEnvironment(Environment):
    """ test time-series environment for artificial data-set"""
    def __init__(self):
        """Initialize environment randomly"""
        self.time=0
        self.action=[]
        self.actionHistory=array([])
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
        currentState=self.worldState[t,:]
        return currentState

    def performAction(self, action):
        """ perform an action on the world that changes it's internal state (maybe
            stochastically).
            :key action: an action that should be executed in the Environment.
            :type action: by default, this is assumed to be a numpy array of doubles
        """
        self.action=action
        self.actionHistory=append(self.actionHistory,action)

    def incrementTime(self):
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

# loads in data from csv file
class MarketEnvironment(TSEnvironment):

    def __init__(self,*args):
        super(MarketEnvironment,self).__init__()
        if len(args)==0:
            inData=self.loadData() #input data as pandas dataframe
            print(type(inData))
        elif len(args)==1:
            inData=args[0]
            print(type(inData))
        else:
            print('something went wrong. The market environment expects at most one argement')
        self.ts=inData['RETURNS'].shift(-1).as_matrix()
        self.worldState=self.createWorldState(inData)
        pass
        #self.ts=self.dataMat[:,0] #just the returns timeseries

    def createWorldState(self,inData):
        # when making a decision at time t, only use data as recent as t-1
        state=inData.shift().ix[1:]
        return state.as_matrix()

    def loadData(self):
        #read in csv file where the dates are the keys
        data=pd.read_csv('data/modelInputs.csv',parse_dates=['DATE'],index_col='DATE')

        #insert a percenage returns column
        data['RETURNS']=data['S&P PRICE'].pct_change()

        #make sure data is complete
        data=data.dropna()
        cols=data.columns.tolist()
        cols=cols[-1:]+cols[:-1]
        data=data[cols]
        data=data.drop('S&P PRICE',1) #don't want the price
        return data



# Special case of AR(1) process

class AR1Environment(TSEnvironment):

    def __init__(self,tsLength):
        super(AR1Environment,self).__init__()
        self.tsLength=tsLength
        self.rho=0.99 #order of autoregression
        self.ts=AR1Environment.__createTS(tsLength,self.rho)
        self.worldState=self.ts[:-1]
        self.ts=self.ts[1:]

    @staticmethod
    def __createTS(tsLength,rho):
        ts=[0 for i in range(tsLength)]
        #ts = array([0.0 for x in range(tsLength)])
        for i in range(1,tsLength):
            ts[i]=rho*ts[i-1]+gauss(0.0,0.2)
        ts=array(ts)
        return ts

# Special case of SnP returns Environment

class DailySnPEnvironment(TSEnvironment):
    def __init__(self):
        super(DailySnPEnvironment,self).__init__()
        self.ts=DailySnPEnvironment.__importSnP()

    @staticmethod
    def __importSnP():
        import csv
        with open('pybrain/rl/environments/timeseries/SnP_data.csv','r') as f:
            data = [row for row in csv.reader(f.read().splitlines())]
        price=[]
        [price.append(data[i][4]) for i in range(1,len(data))]
        price.reverse()
        price=map(float,price)
        rets=matrix([(price[i]-price[i-1])/price[i-1] for i in range(len(price))])
        return rets

class MonthlySnPEnvironment(TSEnvironment):
    def __init__(self):
        super(MonthlySnPEnvironment,self).__init__()
        ts, dates=MonthlySnPEnvironment.__importSnP()
        self.ts=ts
        self.dates=dates

    @staticmethod
    def __importSnP():
        with open('SnP_data.csv','r') as f:
            data = [row for row in csv.reader(f.read().splitlines())]

            data.pop(0) #get rid of the labels
        dates=[]
        price=[]
        dailyLogRets=[]
        monthlyLogRets=[0.0]
        j=0
        for i in range(len(data)):
            price.insert(0,float(data[i][4]))
            dates.insert(0,time.strptime(data[i][0],"%d/%m/%y"))
            if i>0:
                dailyLogRets.insert(0,log(price[1])-log(price[0]))
                if dates[0][1]==dates[1][1]: #if the months are the same
                    monthlyLogRets[0]+=dailyLogRets[0]
                else:
                    monthlyLogRets.insert(0,0)
        return matrix(monthlyLogRets), dates

