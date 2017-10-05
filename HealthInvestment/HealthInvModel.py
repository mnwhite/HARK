'''
This module contains the solver and agent class for the model for "...An Ounce of Prevention..."
'''

import sys
import os
sys.path.insert(0,'../')
sys.path.insert(0,'../ConsumptionSaving/')

from IndShockModel import IndShockConsumerType


class QuadraticFunction(object):
    '''
    A simple class representing a quadratic function.
    '''
    def __init__(self,a0,a1,a2):
        self.a0 = a0
        self.a1 = a1
        self.a2 = a2
    
    def __call__(self,x):
        return self.a0 + self.a1*x + self.a2*x**2


def solveHealthInvestment(solution_next,CRRA,DiscFac,MedCurve,Income,Rfree,Cfloor,LifeUtility,
                          MargUtilityShift,Bequest0,Bequest1,MedPrice,aXtraGrid,bLvlGrid,Hgrid,
                          hGrid,HealthProd0,HealthProd1,HealthProd2,HealthShkStd0,HealthShkStd1,
                          MedShkGrid,ExpHealthNextFunc,ExpHealthNextInvFunc,LivPrbFunc,Gfunc,
                          Dfunc,DpFunc,CritShkFunc,cEffFunc,bFromxFunc,PremiumFunc,CopayFunc,
                          MedShkMeanFunc,MedShkStdFunc):
    '''
    Solves the one period problem in the health investment model.
    
    Parameters
    ----------
    solution_next : HealthInvSolution
        Solution to next period's problem in the same model.  Should have attributes called
        [list of attribute names here].
    CRRA : float
        Coefficient of relative risk aversion.
    DiscFac : float
        Intertemporal discount factor.
    MedCurve : float
        Curvature parameter for mitigative care in utility function.
    Income : float
        Dollars in income received next period.
    Rfree : float
        Gross return factor on end-of-period assets (risk free).
    Cfloor : float
        Effective consumption floor imposed by policy.
    LifeUtility : float
        Additive shifter for utility of being alive.
    MargUtilityShift : float
        Multiplicative shifter on marginal utility as we move from health=0 to health=1.
    Bequest0 : float
        Additive shifter on end-of-life assets in bequest motive function.
    Bequest1 : float
        Magnitude of bequest motive.
    MedPrice : float
        Relative price of a unit of medical care this period (consumption price = 1).
    aXtraGrid : np.array
        Exogenous grid of end-of-period assets for use in the EGM.
    bLvlGrid : np.array
        Exogenous grid of beginning-of-period bank balances for use in the JDfix step.
    Hgrid : np.array
        Exogenous grid of post-investment expected health levels for use in the EGM.
    hGrid : np.array
        Exogenous grid of beginning-of-period health levels for use in the JDfix step.
    MedShkGrid : np.array
        Exogenous grid of medical need shocks.
    HealthProd0 : float
        Curvature of the health production function with respect to investment.
    HealthProd1 : float
        Baseline effectiveness of the health production function.
    HealthProd2 : float
        Change in effectiveness of health production function with illness.
    HealthShkStd0 : float
        Standard deviation of health shocks when in perfect health.
    HealthShkStd1 : float
        Change in stdev of health shocks with illness.
    ExpHealthNextFunc : function
        Expected pre-investment health as a function of current health.
    ExpHealthNextInvFunc : function
        Current health as a function of expected pre-investment health.
    LivPrbFunc : function
        Survival probability to next period as a function of end-of-period health.
    Gfunc : function
        Solution to first order condition for optimal effective consumption as a
        function of pseudo-inverse end-of-period marginal value and medical need.
    Dfunc : function
        Cost of achieving a given level of effective consumption for given medical need.
    DpFunc : function
        Marginal cost of getting a bit more effective consumption.
    CritShkFunc : function
        Critical medical need shock where Cfloor begins to bind as a function of xLvl.
    cEffFunc : function
        Effective consumption achievable as a function of expenditure and medical need.
    bFromxFunc : function
        Transformed consumption share as a function of expenditure and medical need.
    PremiumFunc : function
        Out-of-pocket premiums as a function of health.
    CopayFunc : function
        Out-of-pocket coinsurance rate as a function of health.
    MedShkMeanFunc : function
        Mean of log medical need shock as a function of health.
    MedShkStdFunc : function
        Stdev of log medical need shock as a function of health.
        
    Returns
    -------
    solution_now : HealthInvSolution
        Solution to this period's health investment problem.
    '''
    pass







class HealthInvestmentConsumerType(IndShockConsumerType):
    '''
    A class for representing agents in the health investment model.
    '''
    def __init__(self,**kwargs):
        pass
    
    def updateMedShkDstnFuncs(self):
        '''
        Constructs the attributes MedShkMeanFunc as a time-varying attribute and
        MedShkStdFunc as a time-invarying attribute.
        Each element of these lists is a real to real function that
        takes in a health level and returns a mean or stdev of log medical need.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        None
        '''
        orig_time = self.time_flow
        if not self.time_flow:
            self.timeFwd()
            
        MedShkMeanFunc = []
        MedShkStdFunc = QuadraticFunction(self.MedShkStd0,self.MedShkStd1,0.0)
        for t in range(self.T_cycle):
            beta0 = self.MedShkMean0 + self.MedShkMeanAge*t + self.MedShkMeanAgeSq*t**2
            beta1 = self.MedShkMeanHealth
            beta2 = self.MedShkMeanHealthSq
            MedShkMeanFunc.append(QuadraticFunction(beta0,beta1,beta2))
        
        self.MedShkMeanFunc = MedShkMeanFunc
        self.MedShkStdFunc = MedShkStdFunc
        self.addToTimeVary('MedShkMeanFunc')
        self.addToTimeInv('MedShkStdFunc')
        
        if not orig_time:
            self.timeRev()
        
            
                             