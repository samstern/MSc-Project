from pybrain.rl.environments.timeseries.maximizereturntask import DifferentialSharpeRatioTask
from pybrain.rl.environments.timeseries.timeseries import AR1Environment
from pybrain.optimization import HillClimber
from pybrain.rl.agents import OptimizationAgent
from pybrain.rl.experiments import EpisodicExperiment
from pybrain.structure import RecurrentNetwork
from pybrain.structure import LinearLayer, SigmoidLayer
from pybrain.structure import FullConnection
from pybrain.optimization import PGPE

from matplotlib import pyplot

net= RecurrentNetwork()
#Single linear layer and single sigmoid layer. the linear layer is whats optimised
net.addInputModule(LinearLayer(1,name='in'))
net.addOutputModule(SigmoidLayer(1, name='out'))
net.addConnection(FullConnection(net['in'],net['out'],name='c1'))
net.addRecurrentConnection(FullConnection(net['in'], net['out'], name='c2'))
net.sortModules()

env= AR1Environment(2000)
task=DifferentialSharpeRatioTask(env)
learner = HillClimber(storeAllEvaluations = True)
agent = OptimizationAgent(net,learner)
exp = EpisodicExperiment(task,agent)

numEpisodes = 2
numSamplesPerLearningStep=3

exp.doEpisodes(10)

