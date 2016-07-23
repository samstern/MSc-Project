from pybrain.rl.environments.timeseries.maximizereturntask import MaximizeReturnTask
from pybrain.rl.environments.timeseries.timeseries import RWEnvironment
from pybrain.rl.learners.directsearch.rrl import RRL
from pybrain.structure import RecurrentNetwork
from pybrain.structure import LinearLayer, BiasUnit, FullConnection
from pybrain.structure.modules.signlayer import SignLayer
from pybrain.rl.agents.learning import LearningAgent
from pybrain.rl.experiments import ContinuousExperiment
import performanceEvaluation as pE
import matplotlib.pyplot as plt
import scipy.stats as st
from numpy import cumsum, log, sign, mean
from math import floor

"""
Parameter settings

eta=0.01
step size=0.01

"""
def main():
    numIterations=200
    terminal_EMA_SharpeRatio=[0 for i in range(numIterations)]
    numTrades=[0 for i in range(numIterations)]
    sharpe_first_half=[0 for i in range(numIterations)]
    sharpe_sec_half=[0 for i in range(numIterations)]
    sharpe_ratio_total=[0 for i in range(numIterations)]

    for i in range(numIterations):
        env=RWEnvironment(2000)
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

        ts=env.ts
        learner = RRL(numIn+2,ts) # ENAC() #Q_LinFA(2,1)
        agent = LearningAgent(net,learner)
        exp = ContinuousExperiment(task,agent)

        #performance tracking

        exp.doInteractionsAndLearn(len(ts)-1)
            #print(net._params)
        terminal_EMA_SharpeRatio[i]=learner.ema_sharpeRatio[-1]
        rs=pE.calculateTradingReturn(env.actionHistory,ts)
        sharpe_first_half[i]=pE.annualisedSharpe(rs[:(len(ts)/2)])
        sharpe_sec_half[i]=pE.annualisedSharpe(rs[len(ts)/2:])
        sharpe_ratio_total[i]=pE.annualisedSharpe(rs)
        numTrades[i]=learner.numTrades



    print(net._params)
    print("average number of trades per 1000 observations is {}".format(mean(numTrades)/2))
    print("mean Sharpe ratios are {} with standard errors {}, and {} with standard errors {}".format(mean(sharpe_first_half),st.sem(sharpe_first_half),mean(sharpe_sec_half),st.sem(sharpe_sec_half)))
    print("average sharpe ratio for each entire epoche is {} with standard error {}".format(mean(sharpe_ratio_total),st.sem(sharpe_ratio_total)))
    fig,ax= plt.subplots(nrows=2,ncols=1,sharex=True,sharey=True)
    l1=ax[0].hist(sharpe_first_half,bins=20)
    ax[0].set_title('Annualised Sharpe Ratio (t=0:1000)')
    l2=ax[1].hist(sharpe_sec_half,bins=20)
    ax[1].set_title('Annualised Sharpe Ratio (t=1001:2000)')
    plt.show()


    #plt.hist(numTrades,bins=20)


    #plt.plot(terminal_EMA_SharpeRatio)
    #plt.show()

    actionHist=env.actionHistory
    ts=[t/100 for t in ts]
    cum_log_r=cumsum([log(1+ts[i]) for i in range(len(ts))])
    cum_log_R=cumsum([log(1+(actionHist[i]*ts[i])) for i in range(len(ts))])



    fix, axes = plt.subplots(3, sharex=True)
    ln1=axes[0].plot(cum_log_r,label='Buy and Hold')
    ln2=axes[0].plot(cum_log_R,label='Trading Agent')
    lns=ln1+ln2
    labs=[l.get_label() for l in lns]
    axes[0].legend(lns,labs,loc='upper left')
    axes[0].set_ylabel("Cumulative Log Returns")
    ax[0].set_title("Artificial Series")
    ln3=axes[1].plot(actionHist,'r',label='Trades')
    axes[1].set_ylabel("F(t)")
    axes[2].plot(learner.ema_sharpeRatio)
    axes[2].set_ylabel("EMA Sharpe Ratio")
    plt.show()

if __name__ == '__main__':
    main()