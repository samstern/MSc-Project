from pybrain.rl.environments.timeseries.maximizereturntask import MaximizeReturnTask
from pybrain.rl.environments.timeseries.timeseries import MarketEnvironment
from pybrain.rl.learners.directsearch.rrl import RRL
from pybrain.structure import RecurrentNetwork
from pybrain.structure import LinearLayer, TanhLayer, BiasUnit, FullConnection
from pybrain.structure.modules.signlayer import SignLayer
from pybrain.rl.agents.learning import LearningAgent
from pybrain.rl.experiments import ContinuousExperiment
from pandas import read_csv, DataFrame
import data.functions as fun
import performanceEvaluation as pE
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
    #initialParams=append(array([0.0]),net._params[1:])
    #net._setParameters(initialParams)
    #net._setParameters([ 0.0,-0.05861005,1.64281513,0.98302613])
    #net._setParameters([0., 1.77132063, 1.3843613, 4.73725269])
    #net._setParameters([ 0.0, -0.95173719, 1.92989266, 0.06837472])
    net._setParameters([ 0.0, 1.29560957, -1.14727503, -1.80005888, 0.66351325, 1.19240189])

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
    outData['trading rets']=pE.calculateTradingReturn(outData['Action_Hist'],outData['ts'])
    outData['cum_log_rets']=cumsum([log(1+x) for x in outData['trading rets']])

    paramHist=learner.paramHistory
    plt.figure(0)
    for i in range(len(net._params)):
        plt.plot(paramHist[i])
    plt.draw()

    print(pE.percentOfOutperformedMonths(outData['trading rets'],outData['ts']))


    #ax1.plot(sign(actionHist),'r')
    plt.figure(1)
    outData['cum_log_ts'].plot(secondary_y=True)
    outData['cum_log_rets'].plot(secondary_y=True)
    outData['Action_Hist'].plot()
    plt.draw()
    plt.show()

    #inData['actionHist']=env.actionHist
    #ax2.plot(cumsum([log(1+x) for x in ts]))
    #ax2.plot(cumsum([log(1+(x*sign(y))) for x,y in zip(ts,actionHist)]),'g')


def createDataset():
    data=read_csv('data/data2.csv',parse_dates=['DATE'],index_col='DATE')
    data.drop('DY', axis=1, inplace=True)
    data=data.dropna()
    data['RETURNS']=data['Price'].pct_change()
    rets=DataFrame(data['RETURNS'])*100
    rets['MA10']=fun.sampleMovingAverage(rets,10)
    rets['MA30']=fun.sampleMovingAverage(rets['RETURNS'],50)
    rets['VAR10']=fun.movingVariance(rets['RETURNS'],30)
    rets=rets.dropna()
    return rets

if __name__ == '__main__':
    main()