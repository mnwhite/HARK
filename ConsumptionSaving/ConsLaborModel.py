"""
Consumption-Saving model with Endogenous Labor Supply Model - Intensive Margin
using the endogenous grid method to invert the first order condition.


@author: Tiphanie
"""

import numpy as np
from HARKcore import AgentType, Solution
from HARKutilities import CRRAutility, CRRAutility_inv, CRRAutilityP, CRRAutilityPP, CRRAutilityP_inv, CRRAutility_invP, makeGridExpMult, approxMeanOneLognormal, plotFuncs, combineIndepDstns
from HARKinterpolation import LinearInterp, BilinearInterp, UpperEnvelope

from ConsIndShockModel import IndShockConsumerType, ConsumerSolution, ValueFunc, MargValueFunc 

""" 
Precomputational Step
create fct Chi, Lambda return c and l solution of FOC

"""


class ToddlerConsumerSolution(Solution):
    '''
    A class for representing one period of the solution to a "toddler consumption-
    saving" problem.
    '''
    distance_criteria = ['cFunc']
    
    def __init__(self,cFunc=None,vFunc=None,vPfunc=None):
        if cFunc is not None:
            setattr(self,'cFunc',cFunc)
        if vFunc is not None:
            setattr(self,'vFunc',vFunc)
        if vPfunc is not None:
            setattr(self,'vPfunc',vPfunc)
    
    chi = lambda VpFunc: c
    lamdba = lambda theta:z
    
    """ To complete. solution_now"""
def chiFunc(VpFunc): 
    cNow = ((((1-alpha)/alpha)*W*theta)**alpha)*solution_now 
    
    return c_Now

def lamdbaFunc(theta):
        
    zNow = (alpha/(1-alpha))*cNow/(W*theta)
    lnow = 1 - znow
    if lnow <=0:
        keep
        
    return lnow

             
        
def solveToddlerCSbyEndogenousGrid(solution_next,DiscFac,Rfree,LivPrb, PermGroFac,CRRA,ProdShkDstn,StateGrid):
    '''
    Solves one period of the "toddler consumption-saving" model by using the endogenous
    grid method to invert the first order condition, obviating any search.
    
    Parameters
    ----------
    solution_next : BabyConsumerSolution
        The solution to the next period's problem; should have the attributes
        VpFunc and Cfunc, representing the marginal value and consumption functions.
    DiscFac : float
        Intertemporal discount factor.
    Rfree : float
        Risk free interest rate on assets retained at the end of the period.
    
    PermGroFac : float                                                          **** To check ****
        Expected permanent income growth factor for next period.
    
    CRRA : float
        Coefficient of relative risk aversion.    = RHO
    Utility: u(x)= x^(1-rho)/(1-rho)    
    
    LivPrb : float
        Survival probability; likelihood of being alive at the beginning of
        the succeeding period. 
        
    ProdShkDstn : [np.array]                         -> ** Productivity shocks **
        Distribution of income received next period.  Has three elements, with the
        first a list of probabilities, the second a list of permanent income
        shocks, and the third a list of transitory income shocks.
        
    StateGrid : np.array
        Array of states at which the consumption-saving problem will be solved.
        Represents values of A_t or end-of-period assets.
        
    Returns
    -------
    solution_now : BabyConsumerSolution
        The solution to this period's problem.
    '''
    # Unpack next period's solution and the income distribution, and define the (inverse) marginal utilty function
    vPfunc_next = solution_next.vPfunc
    IncomeProbs = ProdShkDstn[0]
    PermShkVals  = ProdShkDstn[1]
    TranShkVals  = ProdShkDstn[2]
    ShockCount  = IncomeProbs.size
    
    uP = lambda X : CRRAutilityP(X,gam=CRRA)
    uPinv = lambda X : CRRAutilityP_inv(X,gam=CRRA)
    
    # Make tiled versions of the grid of a_t values and the components of the income distribution
    aNowGrid = np.insert(StateGrid,0,0.0) # Add a point at a_t = 0.
    StateCount = aNowGrid.size
    aNowGrid_rep = np.tile(np.reshape(aNowGrid,(StateCount,1)),(1,ShockCount)) # Replicated aNowGrid for each income shock
    PermShkVals_rep = np.tile(np.reshape(PermShkVals,(1,ShockCount)),(StateCount,1)) # Replicated permanent shock values for each a_t state
    TranShkVals_rep = np.tile(np.reshape(TranShkVals,(1,ShockCount)),(StateCount,1)) # Replicated transitory shock values for each a_t state
    IncomeProbs_rep = np.tile(np.reshape(IncomeProbs,(1,ShockCount)),(StateCount,1)) # Replicated shock probabilities for each a_t state
    
    # Find optimal consumption and the endogenous m_t gridpoint for all a_t values
    Reff_array = Rfree/(PermGroFac*PermShkVals_rep) # Effective interest factor on *normalized* end-of-period assets
    mNext = Reff_array*aNowGrid_rep + TranShkVals_rep # Next period's market resources
    vPnext = vPfunc_next(mNext)*PermShkVals_rep**(-CRRA) # Next period's marginal value
    
    """ To check """
    EndOfPeriodvP = DiscFac*Rfree*LivPrb*np.sum(PermGroFac**(-CRRA)*vPnext*IncomeProbs_rep,axis=1)  # Marginal value of end-of-period assets
    
    ''' xNow, qNow'''
    xNowArray = uPinv(EndOfPeriodvP) # Invert the first order condition to find how much we must have *just consumed*
    mNowArray = aNowGrid + qNowArray # Find beginning of period market resources using end-of-period assets and consumption
    """ _ + xNowArray ? """ 

    xNow = (DiscFac*Rfree*LivPrb*EndofPeriodvP)**(-CRRA)
    xNowArray
    
    # Construct consumption and marginal value functions for this period
    xFuncNow = LinearInterp(np.insert(mNowArray,0,0.0),np.insert(xNowArray,0,0.0))
    vPfuncNow = lambda m : uP(xFuncNow(m)) # Use envelope condition to define marginal value
    
    # Make a solution object for this period and return it
    solution_now = ToddlerConsumerSolution(xFunc=xFuncNow,vPfunc=vPfuncNow)
    
    cNowArray = ((((1-alpha)/alpha)*W*theta)**alpha)*solution_now
    zNowArray = (alpha/(1-alpha))*cNow/(W*theta)
    lNowArray = 1-zNowArray
   
    return solution_now
    return cNowArray
    return lNowArray


    
            
class ToddlerConsumerType(IndShockConsumerType):
    '''
    A class for representing an ex ante homogeneous type of consumer in the "toddler
    consumption-saving" model.  These consumers have CRRA utility over current
    consumption and discount future utility exponentially.  Their future income
    is subject to transitory  and permanent shocks, and they can earn gross interest
    on retained assets at a risk free interest factor.  
    
    -> The solution is represented in a normalized way, with all variables divided 
    by permanent income (raised to the appropriate power). 
    
    This model is homothetic in permanent income.
    
    IndShockConsumerType:  A consumer type with idiosyncratic shocks to permanent and transitory income.
      His problem is defined by a sequence of income distributions, survival prob-
    abilities, and permanent income growth rates, as well as time invariant values
    for risk aversion, discount factor, the interest rate, the grid of end-of-
    period assets, and an artificial borrowing constraint.
    '''
    def __init__(self,**kwds):
        AgentType.__init__(self,**kwds)
        self.time_vary = []
        self.time_inv = ['DiscFac','Rfree','PermGroFac','CRRA','IncomeDstn','StateGrid']
        self.pseudo_terminal = False
        self.solveOnePeriod = solveToddlerCSbyEndogenousGrid

        
    def makeIncomeDstn(self):
        '''
        Uses primitive parameters to construct the attribute IncomeDstn.
        '''
        TranShkDstn = approxMeanOneLognormal(N=self.TranShkCount,   # Number of points in discrete approximation
                                            sigma=self.TranShkStd)  # Standard deviation of underlying normal distribution
        PermShkDstn = approxMeanOneLognormal(N=self.PermShkCount,   # Number of points in discrete approximation
                                            sigma=self.PermShkStd)  # Standard deviation of underlying normal distribution
        self.ProdShkDstn = combineIndepDstns(PermShkDstn,TranShkDstn)# Cross the permanent and transitory distributions 
        """ check self.incomedstn or self.prodshkdstn """
        

    def solveTerminal(self): 
        ''' solve for consumption and leisure choice, c and z or chiFunc and lamdbaFunc ? '''
        
        '''
        Solves the terminal period problem, in which the agent will simply
        consume all available resources.  This version simply repacks the
        terminal solution from the baby CS model method and constructs the
        terminal marginal marginal value function.
        '''
        BabyConsumerType.solveTerminal(self)
        temp_soln = self.solution_terminal
        vPPfunc_terminal = lambda x : CRRAutilityPP(x,gam=self.CRRA)
        self.solution_terminal = ToddlerConsumerSolution(cFunc = temp_soln.Cfunc,
                                                         vFunc = temp_soln.Vfunc,
                                                         vPfunc = temp_soln.VpFunc,
                                                         vPPfunc = vPPfunc_terminal)
        self.solution_terminal.HumWlth = 0.0 # No human wealth in terminal period
        self.solution_terminal.MPCmin  = 1.0 # MPC is 1 everywhere in terminal period
    
    def updateSolutionTerminal(self):
        '''
        Updates the terminal period solution for an aggregate shock consumer.
        Only fills in the consumption function and marginal value function.
        
        Parameters
        ----------
        None
            
        Returns
        -------
        None
        '''
        cFunc_terminal  = BilinearInterp(np.array([[0.0,0.0],[1.0,1.0]]),np.array([0.0,1.0]),np.array([0.0,1.0]))
        
        vPfunc_terminal = MargValueFunc2D(cFunc_terminal,self.CRRA)
        mNrmMin_terminal = ConstantFunction(0)
        self.solution_terminal = ConsumerSolution(cFunc=cFunc_terminal,vPfunc=vPfunc_terminal,mNrmMin=mNrmMin_terminal)
        
    
    def postSolve(self):
        '''
        Store the components of the solution as attributes of self for convenience.
        '''
        self.cFunc = [self.solution[t].cFunc for t in range(len(self.solution))]
        self.addToTimeVary('cFunc')
        if hasattr(self.solution[0],'vFunc') and hasattr(self.solution[-1],'vFunc'):
            self.vFunc = [self.solution[t].vFunc for t in range(len(self.solution))]
            self.addToTimeVary('vFunc')
        if hasattr(self.solution[0],'vPfunc') and hasattr(self.solution[-1],'vPfunc'):
            self.vPfunc = [self.solution[t].vPfunc for t in range(len(self.solution))]
            self.addToTimeVary('vPfunc')
        if hasattr(self.solution[0],'vPPfunc') and hasattr(self.solution[-1],'vPPfunc'):
            self.vPPfunc = [self.solution[t].vPPfunc for t in range(len(self.solution))]
            self.addToTimeVary('vPPfunc')
        
            
if __name__ == '__main__':
    from time import clock
    
    # Define dictionaries of example parameter values
                  
    consumer_dict =  {'DiscFac' : 0.95,
                     'Rfree' : 1.03,
                     'PermGroFac' : 1.02,
                     'CRRA' : 2.0,
                     'StateMin' : 0.001,
                     'StateMax' : 20.0,
                     'StateCount' : 16,
                     'PermShkCount' : 9,
                     'TranShkCount' : 9,
                     'PermShkStd' : 0.1,
                     'TranShkStd' : 0.2,
                     'ExponentialGrid' : True
                     }
    
    # Make and solve an example Agent type
    AgentType = ConsumerType(**consumer_dict)
    AgentType.solveOnePeriod = solveToddlerCSbyEndogenousGrid ''' To change, using Endog Grid meth?'''
    AgentType.cycles = 0
    t_start = clock()
    AgentType.solve()
    t_end = clock()
    AgentType.timeFwd()
    print('Solving the consumption-saving model took ' + str(t_end-t_start) + ' seconds.')
    
    # Plot the consumption function in the first period
    plotFuncs(AgentType.cFunc[0],0.0,10.0)
    