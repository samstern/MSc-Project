from pybrain.rl.environments.timeseries.maximizereturntask import MaximizeReturnTask, NeuneierRewardTask
from pybrain.rl.environments.timeseries.timeseries import MarketEnvironment
from pybrain.rl.learners.directsearch.rrl import RRL
from pybrain.structure import RecurrentNetwork
from pybrain.structure import LinearLayer, TanhLayer, BiasUnit, FullConnection
from pybrain.structure.modules.signlayer import SignLayer
from pybrain.rl.agents.learning import LearningAgent
from pybrain.rl.experiments import ContinuousExperiment
from pandas import read_csv, DataFrame
import data.functions as fun
import matplotlib.pyplot as plt
from numpy import log
from numpy import cumsum, sign, array, append

def main():
    inData=createDataset()
    env = MarketEnvironment(inData)
    task = MaximizeReturnTask(env)
    numIn=min(env.worldState.shape)

    net=RecurrentNetwork()
    net.addInputModule(BiasUnit(name='bias'))
    #net.addOutputModule(TanhLayer(1, name='out'))
    net.addOutputModule((SignLayer(1,name='out')))
    net.addRecurrentConnection(FullConnection(net['out'], net['out'], name='c3'))
    net.addInputModule(LinearLayer(numIn,name='in'))
    net.addConnection(FullConnection(net['in'],net['out'],name='c1'))
    net.addConnection((FullConnection(net['bias'],net['out'],name='c2')))
    net.sortModules()
    # remove bias (set weight to 0)
    initialParams=append(array([0.0]),net._params[1:])
    net._setParameters(initialParams)

    #net._setParameters([ 0.0, -0.95173719, 1.92989266, 0.06837472])
    #net._setParameters([ 0.0,-0.57144962, 0.84470806, 0.031639954])

    ts=env.ts
    learner = RRL(numIn+2,ts) # ENAC() #Q_LinFA(2,1)
    agent = LearningAgent(net,learner)
    exp = ContinuousExperiment(task,agent)

    print(net._params)
    exp.doInteractionsAndLearn(len(ts)-1)
    print(net._params)

    outData=DataFrame(inData['RETURNS']/100)
    outData['ts']=[i/100 for i in ts]
    outData['cum_log_ts']=cumsum([log(1+i) for i in outData['ts']])

    outData['Action_Hist']=env.actionHistory
    outData['cum_log_rets']=cumsum([log(1+(x*y)) for x,y in zip(outData['ts'],env.actionHistory)])



    #ax1.plot(sign(actionHist),'r')
    outData['cum_log_ts'].plot(secondary_y=True)
    outData['cum_log_rets'].plot(secondary_y=True)
    outData['Action_Hist'].plot()
    plt.show()

    #inData['actionHist']=env.actionHist
    #ax2.plot(cumsum([log(1+x) for x in ts]))
    #ax2.plot(cumsum([log(1+(x*sign(y))) for x,y in zip(ts,actionHist)]),'g')


def createDataset():
    data=read_csv('data/modelInputs.csv',parse_dates=['DATE'],index_col='DATE')
    data['RETURNS']=data['S&P PRICE'].pct_change()
    rets=DataFrame(data['RETURNS'])*100
    rets['MA10']=fun.sampleMovingAverage(rets,10)
    #rets['VAR10']=fun.movingVariance(rets['RETURNS'],30)
    rets=rets.dropna()
    return rets

if __name__ == '__main__':
    main()