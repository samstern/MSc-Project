from pybrain.rl.environments.timeseries.maximizereturntask import MaximizeReturnTask
from pybrain.rl.environments.timeseries.timeseries import MarketEnvironment
from pybrain.rl.learners.directsearch.rrl import RRL
from pybrain.structure import RecurrentNetwork
from pybrain.structure import LinearLayer, TanhLayer, BiasUnit, FullConnection
from pybrain.structure.modules.signlayer import SignLayer
from pybrain.rl.agents.learning import LearningAgent
from pybrain.rl.experiments import ContinuousExperiment
from pandas import read_csv, DataFrame, Series
import data.functions as fun
import performanceEvaluation as pE
import matplotlib.pyplot as plt
from numpy import insert, append
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


    net._setParameters([ 0.0, 1.29560957, -1.14727503, -1.80005888, 1.0,0.66351325,  1.19240189])
    #net._setParameters([ 1.60523685,  2.40318069,  1.26455845, -0.21875252,  1.35850241, 1.27613626])

    ts=env.ts
    learner = RRL(numIn+2,ts) # ENAC() #Q_LinFA(2,1)
    agent = LearningAgent(net,learner)
    exp = ContinuousExperiment(task,agent)

    in_sample_len=500
    for i in range(1):
        exp.doInteractionsAndLearn(in_sample_len)
        learner.reset()
        agent.reset()
        env.reset()

    print(net._params)
    exp.doInteractionsAndLearn(len(ts)-1)
    print(net._params)


    dfIndex=inData['RETURNS'].index
    outDataAll=pE.outData(ts,env.actionHistory,dfIndex)
    outDataOOS=pE.outData(ts,env.actionHistory,dfIndex,startIndex=in_sample_len)
    #performance evaluation

    rf=0#inData['Fed Fund Target']
    sharpe_oos=pE.annualisedSharpe(outDataOOS['trading rets'],rf)
    drawDown_oos=pE.maximumDrawdown(outDataOOS['trading rets'])
    numOutperformedMonths_oos=pE.percentOfOutperformedMonths(outDataOOS['trading rets'],outDataOOS['ts'])
    print( "oos sharpe: {}, \noos drawdown: {} \noos percent outperformed months {}".format(sharpe_oos, drawDown_oos, numOutperformedMonths_oos))
    print("benchmark sharpe: {}".format(pE.annualisedSharpe(outDataOOS.ts,rf)))
    paramHist=learner.paramHistory
    plt.figure(0)
    inData.rename(columns={'RETURNS': 'r(t-1)'},inplace=True)
    lbs=insert(inData.columns.values,0,'Bias')
    lbs=append(lbs,'F(t-1)')
    print(lbs)
    for i in range(len(net._params)):
        plt.plot(paramHist[i],label=lbs[i])
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1),ncol=3)
    plt.draw()



    #ax1.plot(sign(actionHist),'r')
    fix, axes = plt.subplots(nrows=2,ncols=1)
    plotFrame=outDataOOS[['cum_log_ts','cum_log_rets']]
    plotFrame.columns=['Buy and Hold','Trading Agent']
    plotFrame.plot(ax=axes[0])
    outDataOOS['Action_Hist'].plot(ax=axes[1],color='r')

    plt.draw()
    plt.show()


def createDataset():
    data=read_csv('data/data2.csv',parse_dates=['DATE'],index_col='DATE')
    data=data.dropna()
    data['RETURNS']=data['Price'].pct_change()
    rets=DataFrame(data['RETURNS'])*100
    rets['10d MA']=fun.sampleMovingAverage(rets,10)
    rets['50d MA']=fun.sampleMovingAverage(rets['RETURNS'],50)
    rets['LaggedMA30']=fun.sampleMovingAverage(fun.lag(rets['RETURNS'],30),90)
    rets['1m Var']=fun.movingVariance(rets['RETURNS'],30)
    #rets['LagMA']=fun.lag(fun.sampleMovingAverage(rets['RETURNS'],30),30)
    rets=rets.dropna()
    return rets


if __name__ == '__main__':
    main()