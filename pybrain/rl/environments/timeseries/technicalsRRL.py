from pybrain.rl.environments.timeseries.maximizereturntask import MaximizeReturnTask, NeuneierRewardTask
from pybrain.rl.environments.timeseries.timeseries import MarketEnvironment
from pybrain.rl.learners.directsearch.rrl import RRL
from pybrain.structure import RecurrentNetwork
from pybrain.structure import LinearLayer, TanhLayer, BiasUnit
from pybrain.structure import FullConnection
from pybrain.rl.agents.learning import LearningAgent
from pybrain.rl.experiments import ContinuousExperiment
from pandas import read_csv, DataFrame
import data.functions as fun
import matplotlib.pyplot as plt
from math import log
from numpy import cumsum, sign

def main():
    inData=createDataset()
    env = MarketEnvironment(inData)
    task = MaximizeReturnTask(env)
    numIn=min(env.worldState.shape)

    net=RecurrentNetwork()
    net.addInputModule(BiasUnit(name='bias'))
    net.addOutputModule(TanhLayer(1, name='out'))
    net.addRecurrentConnection(FullConnection(net['out'], net['out'], name='c3'))
    net.addInputModule(LinearLayer(numIn,name='in'))
    net.addConnection(FullConnection(net['in'],net['out'],name='c1'))
    net.addConnection((FullConnection(net['bias'],net['out'],name='c2')))
    net.sortModules()
    #net._setParameters([1.0, 0.0, 100, 2.0]) for eta=0.1
    #net._setParameters([0.5, 0.0, 1.0, 0.0])
    #[ 1.53246823 -1.35962024 -0.75628095 -1.11956953]
    ts=env.ts
    learner = RRL(numIn+2,ts) # ENAC() #Q_LinFA(2,1)
    agent = LearningAgent(net,learner)
    exp = ContinuousExperiment(task,agent)

    print(net._params)
    exp.doInteractionsAndLearn(12000)
    print(net._params)
    actionHist=(env.actionHistory)

    fig, ax1 = plt.subplots()
    ax2=ax1.twinx()

    ax1.plot(cumsum([log(1+x) for x in ts]))
    ax1.plot(cumsum([log(1+(x*sign(y))) for x,y in zip(ts,actionHist)]),'g')
    ax2.plot(actionHist,'r')
    plt.show()

def createDataset():
    data=read_csv('data/modelInputs.csv',parse_dates=['DATE'],index_col='DATE')
    data['RETURNS']=data['S&P PRICE'].pct_change()
    rets=DataFrame(data['RETURNS'])
    rets['MA10']=fun.sampleMovingAverage(rets,50)
    rets=rets.dropna()
    return rets

if __name__ == '__main__':
    main()