from pybrain.rl.environments.timeseries.maximizereturntask import MaximizeReturnTask, NeuneierRewardTask
from pybrain.rl.environments.timeseries.timeseries import AR1Environment
from pybrain.rl.learners.directsearch.rrl import RRL
from pybrain.structure import RecurrentNetwork
from pybrain.structure import LinearLayer, SigmoidLayer, TanhLayer, BiasUnit
from pybrain.structure import FullConnection
from pybrain.rl.agents import LearningAgent
from pybrain.optimization import FiniteDifferences
from pybrain.rl.experiments import ContinuousExperiment

from numpy import sign, round
from matplotlib import pyplot

"""
This script aims to create a trading model that trades on a simple AR(1) process
"""

##building the Recurrent Network

net= RecurrentNetwork()
#Single linear layer with bias unit, and single tanh layer. the linear layer is whats optimised
net.addInputModule(BiasUnit(name='bias'))
net.addOutputModule(TanhLayer(1, name='out'))
net.addRecurrentConnection(FullConnection(net['out'], net['out'], name='c3'))
net.addInputModule(LinearLayer(1,name='in'))
net.addConnection(FullConnection(net['in'],net['out'],name='c1'))
net.addConnection((FullConnection(net['bias'],net['out'],name='c2')))
net.sortModules()
#._setParameters([-0.0, 1.8, 1.6])
print(net._params)
#print(net.activate(0.5))
#print(net.activate(0.6))
#net.activate(2)


env=AR1Environment(2000)
task=MaximizeReturnTask(env)#MaximizeReturnTask(env)
ts=env.ts
learner = RRL(3,ts) # ENAC() #Q_LinFA(2,1)
agent = LearningAgent(net,learner)
exp = ContinuousExperiment(task,agent)

exp.doInteractionsAndLearn(1998)
print(net._params)
actionHist=round(env.actionHistory)
print(actionHist)
pyplot.plot(ts)
pyplot.plot(actionHist)
pyplot.show()


#snp_rets=env.importSnP().tolist()[0]
#print(snp_rets.tolist()[0])
#pyplot.plot(snp_rets)
#pyplot.show()



#cumret= cumsum(multiply(ts,actionHist))


#exp.doInteractions(200)