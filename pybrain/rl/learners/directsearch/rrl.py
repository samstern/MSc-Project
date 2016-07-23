from numpy import tanh, sign
from random import random
from pybrain.rl.learners.directsearch.directsearch import DirectSearchLearner
from pybrain.rl.learners.learner import DataSetLearner, ExploringLearner
from math import sqrt

class RRL(DirectSearchLearner, DataSetLearner, ExploringLearner):
    """Recurrent Reinforcement Learning from Moody and Saffell (2001)"""

    def __init__(self,numParameters,ts):
        self.ts=ts
        self.numParameters=numParameters
        self.num_features=numParameters #TODO: don't have this hardcoded
        self.A=0.0 #moving average returns
        self.B=0.1 #moving average std
        self.delta=0.025 #transaction cost
        self.eta = 0.01
        self.mu = 1.0 # invested capital
        self.rf = 0.0 #risk free rate
        self.stepSize=0.01
        self.F = [-1.0] # last periods output
        self.dFt_dTheta=[0.0 for i in range(numParameters)]
        self.paramUpdateThreshold=0.01 #when to stop ascending the gradient
        self.paramHistory=[[] for i in range(numParameters)] # saving parameters for analysis of results
        self.ema_sharpeRatio=[]
        self.numTrades=0
        #self.explorer=EpsilonGreedyExplorer()

        # create default explorer
        self._explorer = None

    def reset(self):
        self.A=0.0
        self.B=0.1
        self.F=[-1.0]
        self.ema_sharpeRatio=[]
        self.numTrades=0
        self.paramHistory=[[] for i in range(self.numParameters)]


    def explore(self, state, action):
        # forward pass of exploration
        explorative = ExploringLearner.explore(self, state, action)
        return explorative

    def learn(self):
        assert self.dataset != None
        assert self.module != None
        self.thetas=self.module._params.tolist()
        r_t = self.ts[len(self.F)-1]
        #changing notation to that used in the paper by Moody and Saffell. R_t is the reward
        sequenceNumber=self.dataset.getNumSequences()-1
        _state, _lastAction, R_t = self.dataset.getSequence(sequenceNumber)
        R_t = R_t[-1,0] # want it as a list
        #self.F.extend(_lastAction[-1].tolist())
        self.F.append(_lastAction[-1,0])
        phi = [1.0]+_state[-1].tolist()+[sign(self.F[-2])] #TODO: Make sure this still works when _state is more than just 1D
        #phi[0]=0.0 # decided against using a bias




        gradient=self.calculateGradient(R_t,phi,r_t)
        for i in range(len(self.thetas)):
            self.paramHistory[i].append(self.thetas[i])
            self.thetas[i]=self.thetas[i]+self.stepSize*gradient[i]


        self.module._setParameters(self.thetas)

        self.A=self.A+self.eta*(R_t-self.A)
        self.B=self.B+self.eta*((R_t**2)-self.B)

        #performance tracking
        self.ema_sharpeRatio.append(self.A/sqrt(self.B-(self.A**2)))
        if (self.F[-1]!=self.F[-2]):
            self.numTrades+=1
        # TODO: implement parameter update (see policygradient for an example to follow) as well as updating A,B etc.

    def calculateGradient(self,R_t,phi,r_t):
        """phi is the design matrix phi={1,F[t-1],x1,x2,...}
        i is the index of the parameter to update
        R_t is the return of the portfolio at time t
        r_t is the return of the risky asset at time t
        """

        t=len(self.F)-1
        if t==100:
            pass

        F_coef=self.thetas[-1] #the coefficient for F_{t-1}
        new_dFt_dTheta=[0.0 for _ in self.thetas] # initialize temporary dFt_dTheta
        thetaPhi=sum([x*y for x,y in zip(phi,self.thetas)]) #inner product
        dU_dTheta=[0.0 for _ in self.thetas]

        #do the parts that are constant across parameters outside of the loop
        dU_dR=(self.B-(self.A*R_t))/((self.B-(self.A**2))**(3.0/2.0))
        dTanh_dThetaphi=(1-tanh(thetaPhi)**2)
        #dR_dFt=-self.mu*self.delta*sign(sign(self.F[-1])-sign(self.F[-2]))
        dR_dFt=-self.delta*(1+(self.F[-2]*r_t))*sign(self.F[-1]-self.F[-2])
        #dR_dFtMin1=self.mu*(r_t-self.rf)+dR_dFt*(F_coef*dTanh_dThetaphi-1)
        dR_dFtMin1=self.calc_dR_dFtMin1(r_t,dTanh_dThetaphi,F_coef,t)#r_t*(1-self.delta*abs(self.F[-1]-self.F[-2]))+self.delta*(1+(self.F[-2]*r_t))*sign(self.F[-1]-self.F[-2])

        # finding the new dFt_dTheta for each parameter sequentially
        for i in range(len(self.thetas)):
            new_dFt_dTheta[i]=dTanh_dThetaphi*(phi[i]+(F_coef*self.dFt_dTheta[i]))
            #put it all together
            dU_dTheta[i]=dU_dR*((dR_dFt*new_dFt_dTheta[i])+(dR_dFtMin1*self.dFt_dTheta[i]))

        #update dFt_dTheta
        self.dFt_dTheta=new_dFt_dTheta

        return dU_dTheta

    def calc_dR_dFtMin1(self,r_t,dTanh_dThetaphi,F_coef,t):
        a=r_t*(1-self.delta*abs(self.F[-1]-self.F[-2]))
        b=self.delta*(1+self.F[t-1]*r_t)
        c=b*sign(self.F[t-1]-self.F[t-1])
        dFt_dFtMin1=F_coef*dTanh_dThetaphi
        return a+c*dFt_dFtMin1


        # t=len(self.F)-1
        # nu=self.thetas[-1]# the coefficient for F_{t-1}
        # new_dFt_dTheta=[0.0 for _ in self.thetas] # initialize temporary dFt_dTheta
        # thetaPhi=sum([x*y for x,y in zip(phi,self.thetas)]) #inner product
        # #dFt_dTheta=[0.0 for _ in self.thetas]
        # dS_dTheta=[0.0 for _ in self.thetas]
        # #do the parts that are constant across parameters outside of the loop
        # dU_dR=self.eta*(self.B-(self.A*R_t))/((self.B-(self.A**2))**(3.0/2))
        # dFt_dtanh=(1-tanh(thetaPhi)**2)
        #
        # dR_dFt=-self.mu*self.delta*sign(self.F[t]-self.F[t-1])
        # dR_dFtMin1=self.mu*((r_t-self.rf)+(self.delta*(sign(self.F[t]-self.F[t-1])))*(nu*dFt_dtanh-1))
        #
        # #calculating gradient for each parameter
        # for i in range(len(self.thetas)): # for each parameter
        #     new_dFt_dTheta[i]=dFt_dtanh*(phi[i]+nu*self.dFt_dTheta[i])#+self.thetas[i]*self.dFt_dTheta[i])
        #     dS_dTheta[i]=dU_dR*((dR_dFt*new_dFt_dTheta[i])+(dR_dFtMin1*self.dFt_dTheta[i]))
        #
        # self.dFt_dTheta=new_dFt_dTheta
        # self.updateStepSize()
        #
        # return dS_dTheta

    def updateStepSize(self):
        self.stepSize=self.stepSize

