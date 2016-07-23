from pybrain.rl.environments.timeseries.maximizereturntask import MaximizeReturnTask
from pybrain.rl.environments.timeseries.timeseries import MarketEnvironment
from pybrain.rl.learners.directsearch.rrl import RRL
from pybrain.structure import RecurrentNetwork
from pybrain.structure import LinearLayer,BiasUnit, FullConnection
from pybrain.structure.modules.signlayer import SignLayer
from pybrain.rl.agents.learning import LearningAgent
from pybrain.rl.experiments import ContinuousExperiment
from pandas import read_csv, DataFrame, Series
from numpy import insert, append
import math
import data.functions as fun
import performanceEvaluation as pE
import matplotlib.pyplot as plt


def main():
    inData=createDataset()
    env = MarketEnvironment(inData)
    task = MaximizeReturnTask(env)
    numIn=min(env.worldState.shape)

    net=RecurrentNetwork()
    net.addInputModule(BiasUnit(name='bias'))
    net.addOutputModule((SignLayer(1,name='out')))
    net.addRecurrentConnection(FullConnection(net['out'], net['out'], name='c3'))
    net.addInputModule(LinearLayer(numIn,name='in'))
    net.addConnection(FullConnection(net['in'],net['out'],name='c1'))
    net.addConnection((FullConnection(net['bias'],net['out'],name='c2')))
    net.sortModules()
    ###net._setParameters([1.89523389,  2.41243781, -0.37355216, 0.60550426, 1.29560957, -1.14727503, -1.80005888, 0.66351325, 1.91905451])
    ###net._setParameters([ 1.07300605, 2.37801446, -0.28118081, -0.78715898, 0.13367809, 0.31757825,-1.23956247, 1.90411791, 0.95458375])
    ##net._setParameters([1.35840492,1.87785682, -0.15779415, -0.79786631, 0.13380422, 0.0067797, -1.28202562, 2.38574234, 0.909462])
    ###net._setParameters([ 0.36062235, 1.70329005, 2.24180157, 0.34832656, 0.31775365, -0.60400026, -0.44850303, 1.50005529, -0.99986366])
    net._setParameters([ 1.15741417, 1.70427034, 1.05050831, -0.47303435, -0.87220272, -1.44743793,  0.93697461, 2.77489952, 0.27374758])
    ts=env.ts
    learner = RRL(numIn+2,ts) # ENAC() #Q_LinFA(2,1)
    agent = LearningAgent(net,learner)
    exp = ContinuousExperiment(task,agent)


    # in sample learning
    in_sample_len=500
    print("Before in sample {}".format(net._params))
    for i in range(100):
        exp.doInteractionsAndLearn(in_sample_len)
        learner.reset()
        agent.reset()
        env.reset()

    # ouy of sample, online learning
    print("Before oos {}".format(net._params))
    exp.doInteractionsAndLearn(len(ts)-1)
    print("After oos {}".format(net._params))

    #performance evaluation
    dfIndex=inData['RETURNS'].index
    rf=0#inData['Fed Fund Target']
    outDataOOS=pE.outData(ts,env.actionHistory,dfIndex,startIndex=in_sample_len)
    sharpe_oos=pE.annualisedSharpe(outDataOOS['trading rets'],rf)
    drawDown_oos=pE.maximumDrawdown(outDataOOS['trading rets'])
    numOutperformedMonths_oos=pE.percentOfOutperformedMonths(outDataOOS['trading rets'],outDataOOS['ts'])
    foo=outDataOOS['cum_log_rets'][-1]
    bar=math.exp(foo)
    traderReturn=math.exp(outDataOOS['cum_log_rets'][-1])-1
    benchmarkReturn=math.exp(outDataOOS['cum_log_ts'].values[-1])-1
    print( "oos sharpe: {}, \noos drawdown: {} \noos percent outperformed months {}\noos trader return {}".format(sharpe_oos, drawDown_oos, numOutperformedMonths_oos,traderReturn))

    paramHist=learner.paramHistory
    inData.rename(columns={'RETURNS': 'r(t-1)'},inplace=True)
    lbs=insert(inData.columns.values,0,'Bias')
    lbs=append(lbs,'F(t-1)')
    plt.figure(0)
    for i in range(len(net._params)):
        if i<7:
            plt.plot(paramHist[i],label=lbs[i])
        else:
            plt.plot(paramHist[i],'--',label=lbs[i])
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1),ncol=3)
    plt.draw()



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
    rets['long-short']=fun.exponentialMovingAverage((data['Moodys AAA']-data['3 Month'])/10,5)
    rets['short']=fun.exponentialMovingAverage(data['3 Month']/10,10)
    rets['DY']=fun.sampleMovingAverage(data['DY']/10,20).diff()
    rets['10d MA']=fun.sampleMovingAverage(rets['RETURNS'],10)
    rets['30d MA']=fun.sampleMovingAverage(rets['RETURNS'],50)
    #rets['Lagged 3m Momentum']=fun.sampleMovingAverage(fun.lag(rets['RETURNS'],30),90)
    rets['1m Var']=fun.movingVariance(rets['RETURNS'],30)
    #Vix
    vixDF=read_csv('data/vix.csv',parse_dates=['Date'],index_col=['Date'],dtype={'vix' :float})
    vix=Series(vixDF.vix)
    vixVals=vix.values
    vixInd=vix.index
    vixInd=vixInd.order()
    #rets['VIX']=Series(vixVals,vixInd)/10
    rets=rets.dropna()
    return rets


if __name__ == '__main__':
    main()