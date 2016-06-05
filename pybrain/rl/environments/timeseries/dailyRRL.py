from pybrain.rl.environments.timeseries.maximizereturntask import MaximizeReturnTask, NeuneierRewardTask
from pybrain.rl.environments.timeseries.timeseries import MarketEnvironment
from pybrain.rl.learners.directsearch.rrl import RRL
from pybrain.structure import RecurrentNetwork
from pybrain.structure import LinearLayer, TanhLayer, BiasUnit
from pybrain.structure import FullConnection
from pybrain.rl.agents import LearningAgent
from pybrain.rl.experiments import ContinuousExperiment
import matplotlib.pyplot as plt
from math import log
from numpy import cumsum, sign


env = MarketEnvironment()
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
#net._setParameters([-0.1749362, 2.10162725, 0.10726541, 1.67949447, -1.51793343, 2.01329702, 1.57673461])

ts=env.ts
learner = RRL(numIn+2,ts) # ENAC() #Q_LinFA(2,1)
agent = LearningAgent(net,learner)
exp = ContinuousExperiment(task,agent)

exp.doInteractionsAndLearn(10000)
print(net._params)
actionHist=(env.actionHistory)

fig, ax1 = plt.subplots()
ax2=ax1.twinx()

ax1.plot(cumsum([log(1+x) for x in ts]))
ax1.plot(cumsum([log(1+(x*sign(y))) for x,y in zip(ts,actionHist)]),'g')
ax2.plot(actionHist,'r')
plt.show()