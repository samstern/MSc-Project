from numpy import tanh, sign
from random import random
from pybrain.rl.learners.directsearch.directsearch import DirectSearchLearner
from pybrain.rl.learners.learner import DataSetLearner, ExploringLearner
from pybrain.rl.explorers.discrete import EpsilonGreedyExplorer

class RRL(DirectSearchLearner, DataSetLearner, ExploringLearner):
    """Recurrent Reinforcement Learning from Moody and Saffell (2001)"""

    def __init__(self,numParameters,ts):
        self.ts=ts
        self.num_features=numParameters #TODO: don't have this hardcoded
        self.A=0.0 #moving average returns
        self.B=0.1 #moving average std
        self.delta=0.4 #transaction cost
        self.eta = 0.5 #moving average decay parameter
        self.mu = 1.0 # invested capital
        self.rf = 0.0 #risk free rate
        self.stepSize=0.1
        self.F = [0.0] # last periods output
        self.dFt_dTheta=[0.0 for i in range(numParameters)]
        self.paramUpdateThreshold=0.01 #when to stop ascending the gradient
        #self.explorer=EpsilonGreedyExplorer()

        # create default explorer
        self._explorer = None


    def explore(self, state, action):
        # forward pass of exploration
        explorative = ExploringLearner.explore(self, state, sign(action))
        return explorative

    def learn(self):
        assert self.dataset != None
        assert self.module != None
        self.thetas=self.module._params.tolist()
        #changing notation to that used in the paper by Moody and Saffell. R_t is the reward
        sequenceNumber=self.dataset.getNumSequences()-1
        _state, _lastAction, R_t = self.dataset.getSequence(sequenceNumber)
        R_t = R_t[-1,0] # want it as a list
        #self.F.extend(_lastAction[-1].tolist())
        self.F.append(_lastAction[-1,0])
        phi = [1.0]+_state[-1].tolist()+[self.F[-2]] #TODO: Make sure this still works when _state is more than just 1D
        r_t = self.ts[sequenceNumber] # TODO: Make sure that the target timeseries return is always the first row



        gradient=self.calculateGradient(R_t,phi,r_t)
        for i in range(len(self.thetas)):
            self.thetas[i]=self.thetas[i]+self.stepSize*gradient[i]

        self.module._setParameters(self.thetas)

        self.A=self.A+self.eta*(R_t-self.A)
        self.B=self.B+self.eta*((R_t**2)-self.B)

        # TODO: implement parameter update (see policygradient for an example to follow) as well as updating A,B etc.

    def calculateGradient(self,R_t,phi,r_t):
        """phi is the design matrix phi={1,F[t-1],x1,x2,...}
        i is the index of the parameter to update
        R_t is the return of the portfolio at time t
        r_t is the return of the risky asset at time t
        """
        t=len(self.F)-1
        new_dFt_dTheta=[0.0 for _ in self.thetas] # initialize temporary dFt_dTheta
        #dFt_dTheta=[0.0 for _ in self.thetas]
        dS_dTheta=[0.0 for _ in self.thetas]
        #do the parts that are constant across parameters outside of the loop
        dU_dR=self.eta*(self.B-(self.A*R_t))/((self.B-(self.A**2))**(3.0/2))
        dR_dFt=-self.mu*self.delta*sign(self.F[t]-self.F[t-1])
        dR_dFtMin1=self.mu*(r_t-self.rf+self.delta*(sign(self.F[t]-self.F[t-1])))
        thetaPhi=sum([x*y for x,y in zip(phi,self.thetas)])
        #calculating gradient for each parameter
        for i in range(len(self.thetas)): # for each parameter
            new_dFt_dTheta[i]=(1-(tanh(thetaPhi)**2))*(phi[i])#+self.thetas[i]*self.dFt_dTheta[i])
            dS_dTheta[i]=dU_dR*((dR_dFt*new_dFt_dTheta[i])+(dR_dFtMin1*self.dFt_dTheta[i]))

        self.dFt_dTheta=new_dFt_dTheta
        self.updateStepSize()

        return dS_dTheta

    def updateStepSize(self):
        self.stepSize=self.stepSize

