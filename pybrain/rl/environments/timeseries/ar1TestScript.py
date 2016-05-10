from pybrain.rl.environments.timeseries.maximizereturntask import DifferentialSharpeRatioTask
from pybrain.rl.environments.timeseries.timeseries import AR1Environment, SnPEnvironment
from pybrain.rl.learners.valuebased.linearfa import Q_LinFA
from pybrain.rl.agents.linearfa import LinearFA_Agent
from pybrain.rl.experiments import ContinuousExperiment

from matplotlib import pyplot

"""
This script aims to create a trading model that trades on a simple AR(1) process
"""

env=AR1Environment(2000)
task=DifferentialSharpeRatioTask(env)
learner = Q_LinFA(2,1)
agent = LinearFA_Agent(learner)
exp = ContinuousExperiment(task,agent)

from decimal import Decimal
ts=env.ts.tolist()
exp.doInteractionsAndLearn(1999)
actionHist=env.actionHistory
pyplot.plot(ts[0])
pyplot.plot(actionHist)
pyplot.show()
#snp_rets=env.importSnP().tolist()[0]
#print(snp_rets.tolist()[0])
#pyplot.plot(snp_rets)
#pyplot.show()



#cumret= cumsum(multiply(ts,actionHist))


#exp.doInteractions(200)

