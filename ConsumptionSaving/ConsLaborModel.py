#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Consumption-Saving model with Endogenous Labor Supply Model - Intensive Margin
using the endogenous grid method to invert the first order condition.

To solve this agent's problem, we normalized the problem by p_t, the permanent
productivity level, which helps us to eliminate a state variable (p_t).
We can then transform the agent's problem into a cost minimization problem for 
which we solve the effective consumption purchased, x_t= z_t^alpha * c_t
It allows us to solve only one FOC with respect to x_t, instead of 2.

We can then solve the FOC using the Endogenous Gridpoint Method, EGM. 
Faster solution method than the classic rootfinding method but we should keep 
in mind that the EGM solution is not well behaved outside the range of 
gridpoints selected. 

@author: Tiphanie Magne
University of Delaware
"""
import sys 
import os
sys.path.insert(0, os.path.abspath('../'))
sys.path.insert(0, os.path.abspath('./'))

import numpy as np
from HARKcore import AgentType, Solution
from HARKutilities import CRRAutility, CRRAutility_inv, CRRAutilityP, CRRAutilityPP, CRRAutilityP_inv, CRRAutility_invP, \
    makeGridExpMult, approxMeanOneLognormal,addDiscreteOutcomeConstantMean
from HARKinterpolation import LinearInterp,LinearInterpOnInterp1D, VariableLowerBoundFunc2D, BilinearInterp, UpperEnvelope, ConstantFunction
from HARKsimulation import drawDiscrete

from ConsIndShockModel import IndShockConsumerType,PerfForesightConsumerType, ConsumerSolution, \
    ValueFunc, MargValueFunc

from ConsAggShockModel import MargValueFunc2D

from ConsPrefShockModel import PrefShockConsumerType
""" 


"""


class ConsumerLaborSolution(Solution):
    '''
    A class for representing one period of the solution to a Consumer Labor problem.
    '''
    distance_criteria = ['cFunc','LbrFunc']
    
    def __init__(self,cFunc=None,LbrFunc=None,vPfunc=None, bNrmMin = None):
        '''
        The constructor for a new ConsumerSolution object.
        
        Parameters
        ----------
        cFunc : function
            The consumption function for this period, defined over bank balances: c = cFunc(b).
        ***** LbrFunc : function
            The labor function for this period, defined over bank balances: l = LbrFunc(b).
        
        vFunc : function
            The beginning-of-period value function for this period, defined over
            bank balances: v = vFunc(b).
        bNrmMin: 
            
        '''
        if cFunc is not None:
            setattr(self,'cFunc',cFunc)
        if LbrFunc is not None:
            setattr(self,'LbrFunc',LbrFunc)
        if vPfunc is not None:
            setattr(self,'vPfunc',vPfunc)
        if bNrmMin is not None:
            setattr(self,'bNrmMin',bNrmMin)

        
def solveConsLaborIntMarg(solution_next,PermShkDstn,TranShkDstn,LivPrb,DiscFac,CRRA,Rfree,PermGroFac,BoroCnstArt,aXtraGrid,TranShkGrid,vFuncBool,CubicBool,WageRte,LbrCost):
    '''
    Solves one period of the consumption-saving model with endogenous labor supply on the intensive margin
    by using the endogenous grid method to invert the first order condition, obviating any search.
    
    Parameters 
    ----------
    solution_next : ConsumerSolution
        The solution to the next period's problem; should have the attributes
        VpFunc, cFunc and LbrFunc representing the marginal value, consumption and labor functions.
    
    PermShkDstn: [np.array]
        Discrete distribution of permanent productivity shocks. 
    TranShkDstn: [np.array]
        Discrete distribution of transitory productivity shocks.       
    LivPrb : float
        Survival probability; likelihood of being alive at the beginning of
        the succeeding period. 
    DiscFac : float
        Intertemporal discount factor.
    CRRA : float
        Coefficient of relative risk aversion.  
    Rfree : float
        Risk free interest rate on assets retained at the end of the period.
    PermGroFac : float                                                         
        Expected permanent income growth factor for next period.
    BoroCnstArt: float or None
        Borrowing constraint for the minimum allowable assets to end the
        period with.  If it is less than the natural borrowing constraint,
        then it is irrelevant; BoroCnstArt=None indicates no artificial bor-
        rowing constraint.
    aXtraGrid: np.array
            Array of "extra" end-of-period asset values-- assets above the
            absolute minimum acceptable level.
    TranShkGrid: np.array
            Array of transitory shock values.
    vFuncBool: boolean
        An indicator for whether the value function should be computed and
        included in the reported solution.
    CubicBool: boolean
        An indicator for whether the solver should use cubic or linear inter-
        polation.
    WageRte: float
        Wage rate
    LbrCost: float
        alpha parameter indicating labor cost.
    
        
    Returns
    -------
    solution_now : ConsumerSolution
        The solution to this period's problem.
    '''
    
    frac = 1/(1+LbrCost)
    if CRRA < frac*LbrCost:
        raise NotImplementedError()
    if BoroCnstArt is not None:
        raise NotImplementedError()        
    if vFuncBool or CubicBool is True:
        raise NotImplementedError()
        
    '''** 2/ To check **'''
    # Unpack next period's solution and the productivity shock distribution, and define the (inverse) marginal utilty function
    vPfunc_next = solution_next.vPfunc  
    TranShkPrbs = TranShkDstn[0]    
    TranShkVals  = TranShkDstn[1]
    PermShkPrbs = PermShkDstn[0]
    PermShkVals  = PermShkDstn[1]         
    ShockCount  = TranShkDstn.size
   
#    uP = lambda X : CRRAutilityP(X,gam=CRRA)
    uPinv = lambda X : CRRAutilityP_inv(X,gam=CRRA)
    uinv = lambda X : CRRAutility_inv(X,gam=CRRA)

    # Make tiled versions of the grid of a_t values and the components of the shock distribution
    aXtraCount = aXtraGrid.size
    aXtraGrid_rep = np.tile(np.reshape(aXtraGrid,(aXtraCount,1)),(1,ShockCount)) # Replicated bNowGrid for each productivity shock
    PermShkVals_rep = np.tile(np.reshape(PermShkVals,(1,ShockCount)),(aXtraCount,1)) # Replicated permanent shock values for each b_t state   
    TranShkGrid = np.tile(np.reshape(TranShkVals,(1,ShockCount)),(aXtraCount,1)) # Replicated transitory shock values for each b_t state
    TranShkPrbs_rep = np.tile(np.reshape(TranShkPrbs,(1,ShockCount)),(aXtraCount,1)) # Replicated transitory shock probabilities for each b_t state
    
    # **** Find optimal consumption + leisure and the endogenous b_t gridpoint for all a_t values ** To precise
  
    bNext = (Rfree*aXtraGrid_rep)/(PermGroFac*PermShkVals_rep)   # Next period's bank balances ~ market resources
    vPNext = vPfunc_next(bNext, TranShkGrid) # Derive the Next period's marginal value from the marginal utility function
    vPbarNext = np.sum(vPNext*TranShkPrbs_rep, axis = 1) # Integrate out the transitory shocks, since the shocks are serially uncorrelated
    
    vPbarNvrsNext = uPinv(vPbarNext) # Invert the marginal utility function
    vPbarNvrsFuncNext = LinearInterp(np.insert(aXtraGrid,0,0.0),np.insert(vPbarNvrsNext,0,0.0)) # Linear interpolation over the b_t. Add a point at b_t = 0.
    vPbarFuncNext = MargValueFunc(vPbarNvrsFuncNext,CRRA) # Take the marginal utility function to inverse back and get the optimal values for consumption
  
    
    ''' 3/ '''
    
    EndOfPrdvP = DiscFac*Rfree*LivPrb*np.sum((PermGroFac*PermShkVals_rep)**(-CRRA)*vPbarFuncNext,axis=1)  # Marginal value of end-of-period assets

    TranShkScaleFac = frac*WageRte*TranShkGrid**(LbrCost*frac)*(LbrCost**(-LbrCost*frac)+LbrCost**(frac)) # ***
    
    ''' 4/ '''
    xArray = (TranShkScaleFac*(EndOfPrdvP))**(-1/(CRRA-LbrCost*frac))    # Get an array of x_t values corresponding to (a_t,theta_t) values
    xNowArray = uinv(xArray)   # Invert the FOC to find how much we must have as effective consumption x_t
    
    ''' 5/'''
    cNrmNow = (((WageRte*TranShkDstn)/LbrCost)**(LbrCost*frac))*xNowArray**frac # Find optimal consumption using the solution to the "composite cons" pb
    LsrNow = (LbrCost/(WageRte*TranShkDstn)**frac)*xNowArray**frac  # Find optimal leisure amount using the solution to the "composite cons" pb
    
    if LsrNow > 1:
        cNrmNow = EndOfPrdvP**(-1/CRRA)
        LsrNow =1
    return cNrmNow
    LbrNow = 1 -LsrNow
    return LbrNow
    
    '''6/'''
    bNrmNow = np.tile(np.reshape(aXtraGrid,TranShkGrid,(1,cNrmNow)),(LsrNow,1))       # Create array of beginning of period bank balances
    bNowArray = aXtraGrid + LsrNow*(1-TranShkGrid) + cNrmNow    # Find beginning of period bank balances using end-of-period assets and produtivity shocks
    bNowExtra = -WageRte*TranShkDstn
    bNowArrayAlt = np.concatenate(bNowExtra, bNowArray) # Combine the two pieces of the b_t grid (when c_t=0, z_t=0 and following grid)
    '''7'''
    vPnvrsNow = EndOfPrdvP**(-1/CRRA)
    vPnvrsNowArray = np.concatenate(np.zeros_like(vPnvrsNow), vPnvrsNow)
    
    # Construct consumption and marginal value functions for this period
    '''8/ '''
    bNrmMinNow = LinearInterp(np.insert(bNowArrayAlt,0,0.0),np.insert(TranShkGrid,0,0.0))
    ''' 9/ '''
    # Make a linear interpolation to get lists of optimal consumption, labor and (pseudo-inverse) marginal value, for each row of theta_t values.
    cFuncNow_list   = []
    LbrFuncNow_list   = []
    vPnvrsFuncNow_list   = []
    for j in range(TranShkGrid.size):
        cFuncNow = LinearInterp(np.insert(cNrmNow,0,0.0), np.insert(bNrmNow[j,:]))
        cFuncNow_list.append(cFuncNow)
        
        LbrFuncNow = LinearInterp(np.insert(LbrNow,0,0.0), np.insert(bNrmNow[j,:]))
        LbrFuncNow_list.append(LbrFuncNow)
        
        vPnvrsFuncNow = LinearInterp(np.insert(vPnvrsNowArray,0,0.0), np.insert(bNrmNow[j,:]))
        vPnvrsFuncNow_list.append(vPnvrsFuncNow)
    '''10'''
    # Make linear interpolation by combining the lists of consumption, labor and value functions
    cFuncNowBase = LinearInterpOnInterp1D(cFuncNow_list,TranShkGrid)
    LbrFuncNowBase = LinearInterpOnInterp1D(LbrFuncNow_list,TranShkGrid)
    vPnvrsFuncNowBase = LinearInterpOnInterp1D(vPnvrsFuncNow_list,TranShkGrid)    
    '''11'''
    cFuncNow  = VariableLowerBoundFunc2D(cFuncNowBase,bNrmMinNow)
    LbrFuncNow=  VariableLowerBoundFunc2D(LbrFuncNowBase,bNrmMinNow)
    vPnvrsFuncNow =  VariableLowerBoundFunc2D(vPnvrsFuncNowBase,bNrmMinNow)      
    '''12'''
    vPfuncNow = MargValueFunc2D(vPnvrsFuncNow,CRRA)   # Construct the marginal value function using the envelope condition
    '''13'''
    # Make a solution object for this period and return it
    solution_now = ConsumerLaborSolution(cFunc=cFuncNow,LbrFunc=LbrFuncNow, vPfunc=vPfuncNow,bNrmMin=bNrmMinNow)
    return solution_now

    
class LaborIntMargConsumerType(IndShockConsumerType):
    
    '''        
    A class for representing an ex ante homogeneous type of consumer in the
    consumption-saving model.  These consumers have CRRA utility over current
    consumption and discount future utility exponentially.  Their future income
    is subject to transitory  and permanent shocks, and they can earn gross interest
    on retained assets at a risk free interest factor.  
    
    The solution is represented in a normalized way, with all variables divided 
    by permanent income (raised to the appropriate power). 
    
    This model is homothetic in permanent income.
    
    IndShockConsumerType:  A consumer type with idiosyncratic shocks to permanent and transitory income.
    His problem is defined by a sequence of income distributions, survival probabilities, 
    and permanent income growth rates, as well as time invariant values for risk aversion, 
    discount factor, the interest rate, the grid of end-of-period assets, and an artificial borrowing constraint.
    '''

    def __init__(self,**kwds):
        '''
        Instantiate a new consumer type with given data.
        See ConsumerParameters.init_labor_intensive for a dictionary of
        the keywords that should be passed to the constructor.
        
        Parameters
        ----------
        cycles : int
            Number of times the sequence of periods should be solved.
        time_flow : boolean
            Whether time is currently "flowing" forward for this instance.
        
        Returns
        -------
        None
    '''  
        IndShockConsumerType.__init__(self,cycles=1,time_flow=True,**kwds)
        
        self.time_vary = []
        self.time_inv = ['DiscFac','Rfree','PermGroFac','CRRA','LbrCost']

        self.pseudo_terminal = False
        self.solveOnePeriod = solveConsLaborIntMarg
        self.TranShkGrid()
        
     
 
    def calcBoundingValues(self):      
        '''
        Calculate human wealth plus minimum and maximum MPC in an infinite
        horizon model with only one period repeated indefinitely.  Store results
        as attributes of self.  Human wealth is the present discounted value of
        expected future income after receiving income this period, ignoring mort-
        ality.  The maximum MPC is the limit of the MPC as m --> mNrmMin.  The
        minimum MPC is the limit of the MPC as m --> infty.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        None
        '''
        raise NotImplementedError()
        
    def makeEulerErrorFunc(self,mMax=100,approx_inc_dstn=True):
        '''
        Creates a "normalized Euler error" function for this instance, mapping
        from market resources to "consumption error per dollar of consumption."
        Stores result in attribute eulerErrorFunc as an interpolated function.
        Has option to use approximate income distribution stored in self.IncomeDstn
        or to use a (temporary) very dense approximation.
        
        NOT YET IMPLEMENTED FOR THIS CLASS
        
        Parameters
        ----------
        mMax : float
            Maximum normalized market resources for the Euler error function.
        approx_inc_dstn : Boolean
            Indicator for whether to use the approximate discrete income distri-
            bution stored in self.IncomeDstn[0], or to use a very accurate
            discrete approximation instead.  When True, uses approximation in
            IncomeDstn; when False, makes and uses a very dense approximation.
        
        Returns
        -------
        None
        '''
        raise NotImplementedError()
    
    def updateSolutionTerminal(self):
        ''' 
        Updates the terminal period solution and solves for optimal consumption and labor when there is no future.
        
        
        Parameters
        ----------
        None
            
        Returns
        -------
        None
        '''
        
        bNrmNow = self.aXtraGrid
        
        self.ShockCount  = len(self.TranShkDstn)
        self.aXtraCount = len(self.aXtraGrid)
        self.aXtraGrid_rep = np.tile(np.reshape(self.aXtraGrid,(self.aXtraCount,1)),(1,self.ShockCount)) # Replicated bNowGrid for each productivity shock
        self.TranShkGrid = np.tile(np.reshape(self.TranShkVals,(1,self.ShockCount)),(self.aXtraCount,1))
        
        cFunc_terminal  = (1/(1+self.LbrCost))*(bNrmNow+self.WageRte*self.TranShkGrid[-1])
        LbrFunc_terminal = (self.LbrCost/(1+self.LbrCost))*(bNrmNow/(self.WageRte*self.TranShkGrid[-1])+1)
        
        
        x_eff_1 = LbrFunc_terminal ** self.LbrCost    
        x_eff = [a*b for a,b in zip(x_eff_1,cFunc_terminal)]
                
        if LbrFunc_terminal.any() == 1:
            LbrFunc_terminal = cFunc_terminal
        if LbrFunc_terminal.any() > 1:
            raise NotImplementedError()
        
        
        vFunc_terminal = CRRAutility(x_eff,gam=self.CRRA)
        vPfunc_terminal = MargValueFunc2D(cFunc_terminal,self.CRRA)
        mNrmMin_terminal = ConstantFunction(0)
        self.solution_terminal = ConsumerSolution(cFunc=cFunc_terminal,LbrFunc=LbrFunc_terminal, vFunc= vFunc_terminal, \
                                                  vPfunc=vPfunc_terminal,mNrmMin=mNrmMin_terminal)
        

    def updateTranShkGrid(self):
        TranShkGridMin  = self.solution[0].TranShkDstn + 10**(-15) # add tiny bit to get around 0/0 problem
        TranShkGridMax = np.max(self.TranShkDstn)
        TranShkGridCount = self.TranShkDstn.size
        TranShkGrid = np.linspace(TranShkGridMin, TranShkGridMax, TranShkGridCount)

###############################################################################
          
if __name__ == '__main__':
    import ConsumerParametersTM as Params
    import matplotlib.pyplot as plt
    from HARKutilities import plotFuncs
    from time import clock
    mystr = lambda number : "{:.4f}".format(number)
    
    do_simulation = False
    
    # Make and solve a labor intensive margin consumer i.e. a consumer with utility for leisure
    LaborIntMargExample = LaborIntMargConsumerType(**Params.init_labor_intensive)
    LaborIntMargExample.cycles = 0 # Infinite horizon 
    
    t_start = clock()
    LaborIntMargExample.solve()
    t_end = clock()
    print('Solving a labor intensive margin consumer took ' + str(t_end-t_start) + ' seconds.')
    LaborIntMargExample.unpackcFunc()
    LaborIntMargExample.unpackLbrFunc() # **** check how it can be done
    
    LaborIntMargExample.timeFwd() # Set the direction of time to ordinary chronological order

                  

    
    # Plot the consumption and labor functions in the first period
    print('Consumption function:')
    mMin = LaborIntMargExample.solution[0].mNrmMin
    plotFuncs(LaborIntMargExample.cFunc[0],mMin,mMin+10)
    
    print('Labor function:')
    mMin = LaborIntMargExample.solution[0].mNrmMin
    plotFuncs(LaborIntMargExample.LbrFunc[0],mMin,mMin+10)

    
