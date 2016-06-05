from pybrain.rl.environments.timeseries.maximizereturntask import MaximizeReturnTask
from pybrain.rl.environments.timeseries.timeseries import MonthlySnPEnvironment
from pybrain.rl.learners.directsearch.rrl import RRL

from pybrain.structure import RecurrentNetwork
from pybrain.structure import LinearLayer, SigmoidLayer, TanhLayer, BiasUnit
from pybrain.structure import FullConnection
from pybrain.rl.agents import LearningAgent
from pybrain.rl.experiments import ContinuousExperiment

from numpy import sign, round
from matplotlib import pyplot

net= RecurrentNetwork()
#Single linear layer with bias unit, and single tanh layer. the linear layer is whats optimised
net.addInputModule(BiasUnit(name='bias'))
net.addOutputModule(TanhLayer(1, name='out'))
net.addRecurrentConnection(FullConnection(net['out'], net['out'], name='c3'))
net.addInputModule(LinearLayer(1,name='in'))
net.addConnection(FullConnection(net['in'],net['out'],name='c1'))
net.addConnection((FullConnection(net['bias'],net['out'],name='c2')))
net.sortModules()
net._setParameters([0, 10, 1.259])
print(net._params)
env=MonthlySnPEnvironment()
task=MaximizeReturnTask(env)
learner = RRL() # ENAC() #Q_LinFA(2,1)
agent = LearningAgent(net,learner)
exp = ContinuousExperiment(task,agent)

ts=env.ts.tolist()
exp.doInteractionsAndLearn(795)
print(net._params)
actionHist=sign(env.actionHistory)/20
pyplot.plot(ts[0])
pyplot.plot(actionHist)
pyplot.show()



######################


from matplotlib import pyplot


#learner = HillClimber(storeAllEvaluations = True)
#agent = OptimizationAgent(net,learner)
#exp = EpisodicExperiment(task,agent)

#numEpisodes = 2
#numSamplesPerLearningStep=3

#exp.doEpisodes(10)

