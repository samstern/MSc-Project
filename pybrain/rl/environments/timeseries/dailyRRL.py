from pybrain.rl.environments.timeseries.maximizereturntask import MaximizeReturnTask, NeuneierRewardTask
from pybrain.rl.environments.timeseries.timeseries import MarketEnvironment
from pybrain.rl.learners.directsearch.rrl import RRL
from pybrain.structure import RecurrentNetwork
from pybrain.structure import LinearLayer, TanhLayer, BiasUnit
from pybrain.structure import FullConnection
from pybrain.rl.agents import LearningAgent
from pybrain.rl.experiments import ContinuousExperiment
from matplotlib import pyplot

env = MarketEnvironment()
task = MaximizeReturnTask(env)
numIn=len(env.dataFrame.columns)

net=RecurrentNetwork()
net.addInputModule(BiasUnit(name='bias'))
net.addOutputModule(TanhLayer(1, name='out'))
net.addRecurrentConnection(FullConnection(net['out'], net['out'], name='c3'))
net.addInputModule(LinearLayer(numIn,name='in'))
net.addConnection(FullConnection(net['in'],net['out'],name='c1'))
net.addConnection((FullConnection(net['bias'],net['out'],name='c2')))
net.sortModules()
learner = RRL(numIn+2) # ENAC() #Q_LinFA(2,1)
agent = LearningAgent(net,learner)
exp = ContinuousExperiment(task,agent)
ts=env.ts[:,0]
exp.doInteractionsAndLearn(10000)
print(net._params)
actionHist=(env.actionHistory)
print(actionHist)
pyplot.plot(ts)
pyplot.plot(actionHist)
pyplot.show()