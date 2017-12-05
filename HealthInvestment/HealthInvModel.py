'''
This module contains the solver and agent class for the model for "...An Ounce of Prevention..."
'''

import sys
import os
sys.path.insert(0,'../')
sys.path.insert(0,'../ConsumptionSaving/')

from copy import copy
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
from HARKcore import NullFunc, Solution, HARKobject, AgentType
from HARKinterpolation import ConstantFunction, LinearInterp, BilinearInterp, TrilinearInterp, LinearInterpOnInterp1D
from HARKutilities import makeGridExpMult, CRRAutility, CRRAutilityP, CRRAutilityP_inv, CRRAutility_inv
from HARKsimulation import drawNormal
from ConsIndShockModel import IndShockConsumerType, ValueFunc
from ConsAggShockModel import MargValueFunc2D
from JorgensenDruedahl3D import JDfixer
import matplotlib.pyplot as plt


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
    
    def der(self,x):
        return self.a1 + 2*self.a2*x
    
    def inverse(self,x):
        c = self.a0 - x
        b = self.a1
        a = self.a2
        discrim_arg = b**2 - 4.*a*c
        discrim_arg[discrim_arg < 0.] = np.nan
        discrim = np.sqrt(discrim_arg)
        if a < 0.:
            out = (-b - discrim)/(2.*a)
        elif a > 0.:
            out = (-b + discrim)/(2.*a)
        else:
            out = -c/b
        return out
    
    
class ValueFunc2D(HARKobject):
    '''
    A class for representing a value function in models with two state variables.
    '''
    distance_criteria = ['vNvrsFunc','CRRA','vLim']
    
    def __init__(self,vNvrsFunc,CRRA,vLim=0.):
        '''
        Constructor for a new value function object.
        
        Parameters
        ----------
        vNvrsFunc : function
            A real function representing the value function composed with the inverse
            utility function, defined on normalized individual market resources and
            beginning of period health.
        CRRA : float
            Coefficient of relative risk aversion.
        vLim : float
            Value that the value function asymptotes to (or at least never reaches)
            as market resources go to infinity.
            
        Returns
        -------
        None
        '''
        self.vNvrsFunc = vNvrsFunc
        self.CRRA = CRRA
        self.vLim = vLim
        
    def __call__(self,b,h):
        return CRRAutility(self.vNvrsFunc(b,h),gam=self.CRRA) + self.vLim
    
    
class LogOnLogFunc2D(HARKobject):
    '''
    A class for 2D interpolated functions in which both the first argument and
    the output are transformed through the natural log function.  This tends to
    smooth out functions whose gridpoints have disparate spacing.  Takes as its
    only argument a 2D function (whose x-domain and range are both R_+).
    '''
    distance_criteria = ['func']
    
    def __init__(self,func):
        self.func = func
        
    def __call__(self,x,y):
        return np.exp(self.func(np.log(x),y))
    
    def derivativeX(self,x,y):
        return self.func.derivativeX(np.log(x),y)/x*self.__call__(x,y)
    
    def derivativeY(self,x,y):
        return self.func.derivativeY(np.log(x),y)*self.__call__(x,y)
    
    
class MargCostFunc(HARKobject):
    '''
    A class for representing "marginal cost of achieving effective consumption"
    functions.  These functions take in a level of effective consumption and a
    medical need shock and return the marginal cost of effective consumption at
    this level when facing this shock.  This representation distorts the cEff
    input by log and the marginal cost by -log(Dp-1) when storing the interpolated
    function.  This class' __call__ function returns the de-distored marg cost.
    '''
    distance_criteria = ['func']
    
    def __init__(self,func):
        self.func = func
        
    def __call__(self,x,y):
        return np.exp(-self.func(np.log(x),y)) + 1.0
    
    
class HealthInvestmentPolicyFunc(HARKobject):
    '''
    A class representing the policy function for the health investment model.
    Its call function returns cLvl, MedLvl, iLvl.  Also has functions for these
    controls individually, and for xLvl.
    '''
    def __init__(self, xFunc, iFunc, bFromxFunc, CopayFunc):
        '''
        Constructor method for a new instance of HealthInvestmentPolicyFunc.
        
        Parameters
        ----------
        xFunc : function
            Expenditure function (cLvl & MedLvl), defined over (bLvl,hLvl,MedShk).
        iFunc : function
            Health investment function, defined over (bLvl,hLvl,MedShk).
        bFromxFunc : function
            Transformed consumption share function, defined over (xLvl,MedShkAdj).  Badly named.
        CopayFunc : function
            Coinsurance rate for medical care as a function of hLvl.
        
        Returns
        -------
        None
        '''
        self.xFunc = xFunc
        self.iFunc = iFunc
        self.bFromxFunc = bFromxFunc
        self.CopayFunc = CopayFunc
        
    def __call__(self,bLvl,hLvl,MedShk):
        '''
        Evaluates the policy function and returns all three controls.
        
        Parameters
        ----------
        bLvl : np.array
            Array of bank balance values.
        hLvl : np.array
            Array of health levels.
        MedShk : np.array
            Array of medical need shocks.
            
        Returns
        -------
        cLvl : np.array
            Array of consumption levels.
        MedLvl : np.array
            Array of mitigative medical care levels.
        iLvl : np.array
            Array of health investment levels.
        '''
        xLvl = self.xFunc(bLvl,hLvl,MedShk)
        Copay = self.CopayFunc(hLvl)
        cShareTrans = self.bFromxFunc(xLvl,MedShk*Copay)
        q = np.exp(-cShareTrans)
        cLvl = xLvl/(1.+q)
        MedLvl = xLvl*q/(1.+q)
        iLvl = self.iFunc(bLvl,hLvl,MedShk)
        return cLvl, MedLvl, iLvl, xLvl
    
    
    def cFunc(self,bLvl,hLvl,MedShk):
        cLvl, MedLvl, iLvl, xLvl = self(bLvl,hLvl,MedShk)
        return cLvl
    
    def MedFunc(self,bLvl,hLvl,MedShk):
        cLvl, MedLvl, iLvl, xLvl = self(bLvl,hLvl,MedShk)
        return MedLvl
    
    def iFunc(self,bLvl,hLvl,MedShk):
        cLvl, MedLvl, iLvl, xLvl = self(bLvl,hLvl,MedShk)
        return iLvl
    
    def xFunc(self,bLvl,hLvl,MedShk):
        cLvl, MedLvl, iLvl, xLvl = self(bLvl,hLvl,MedShk)
        return xLvl
    
    
    
class HealthInvestmentSolution(Solution):
    '''
    A class representing the solution of a single period of a health investment
    problem.  The solution must include an expenditure function and marginal
    value function.
    
    Here and elsewhere in the code, Nrm indicates that variables are normalized
    by permanent income.
    '''
    distance_criteria = ['vPfunc']
    
    def __init__(self, PolicyFunc=None, vFunc=None, dvdbFunc=None, dvdhFunc=None):
        '''
        The constructor for a new ConsumerSolution object.
        
        Parameters
        ----------
        PolicyFunc : HealthInvestmentPolicyFunc
            Optimal behavior this period as a function of market resources, health,
            and the medical need shock.  Returns cLvl, MedLvl, InvLvl when called.
        vFunc : function
            The beginning-of-period value function for this period, defined over
            market resources, health, and medical need: v = vFunc(bLvl,hLvl,MedShk).
        dvdbFunc : function
            The beginning-of-period marginal value with respect to market resources
            function for this period, defined over market resources, health, and
            medical need: dv/db = dvdbFunc(bLvl,hLvl,MedShk).
        dvdhFunc : function
            The beginning-of-period marginal value with respect to health level
            function for this period, defined over market resources, health, and
            medical need: dv/dh = dvdhFunc(bLvl,hLvl,MedShk).
            
        Returns
        -------
        None        
        '''
        # Change any missing function inputs to NullFunc
        if PolicyFunc is None:
            PolicyFunc = NullFunc()
        if vFunc is None:
            vFunc = NullFunc()
        if dvdbFunc is None:
            dvdbFunc = NullFunc()
        if dvdhFunc is None:
            dvdhFunc = NullFunc()
        self.PolicyFunc   = PolicyFunc
        self.vFunc        = vFunc
        self.dvdbFunc     = dvdbFunc
        self.dvdhFunc     = dvdhFunc
        
        
def makebFromxFunc(xLvlGrid,MedShkGrid,CRRA,MedCurve):
    '''
    Constructs a function that returns transformed consumption share as a function
    of total expenditure xLvl and the (price-modified) medical need shock MedShk.
    
    Parameters
    ----------
    xLvlGrid : np.array
        1D array of total expenditure levels.
    MedShkGrid : np.array
        1D array of medical shocks.
    CRRA : float
        Coefficient of relative risk aversion for effective consumption.
    MedCurve : float
        Curvature parameter for medical care in the utility function.
        
    Returns
    -------
    bFunc : BilinearInterp
        Transformed consumption share as a function of total expenditure xLvl and
        the (price-modified) medical need shock MedShk.
    '''
    # Calculate optimal transformed consumption/spending ratio at each combination of mLvl and MedShk.
    bGrid = np.zeros((xLvlGrid.size,MedShkGrid.size)) # Initialize consumption grid
    tempf = lambda b : np.exp(-b)
    for i in range(xLvlGrid.size):
        xLvl = xLvlGrid[i]
        for j in range(MedShkGrid.size):
            MedShk = MedShkGrid[j]
            if xLvl == 0: 
                bOpt = 0.0  # Split 50-50, which still makes cLvl=0
            elif MedShk == 0.0: 
                bOpt = np.nan # Placeholder for when MedShk = 0
            else:
                optMedZeroFunc = lambda q : 1. + MedCurve/MedShk*xLvl/(1.+q) - np.exp((xLvl*q/(1.+q))/MedShk)
                optFuncTransformed = lambda b : optMedZeroFunc(tempf(b))
                bOpt = brentq(optFuncTransformed,-100.0,100.0)
            bGrid[i,j] = bOpt

    # Fill in the missing values
    bGridMax = np.nanmax(bGrid,axis=1)
    for i in range(xLvlGrid.size):
        these = np.isnan(bGrid[i,:])
        bGrid[i,these] = bGridMax[i] + 3.
        
    bFunc = BilinearInterp(bGrid,xLvlGrid,MedShkGrid)
    return bFunc
        


def solveHealthInvestment(solution_next,CRRA,DiscFac,MedCurve,IncomeNext,IncomeNow,Rfree,Cfloor,LifeUtility,
                          MargUtilityShift,Bequest0,Bequest1,MedPrice,aXtraGrid,bLvlGrid,Hgrid,
                          hLvlGrid,HealthProdFunc,MargHealthProdInvFunc,HealthShkStd0,HealthShkStd1,
                          MedShkCount,ExpHealthNextFunc,ExpHealthNextInvFunc,LivPrbFunc,Gfunc,
                          Dfunc,DpFunc,CritShkFunc,cEffFunc,bFromxFunc,PremiumFunc,CopayFunc,
                          MedShkMeanFunc,MedShkStdFunc,ConvexityFixer):
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
    IncomeNext : float
        Dollars in income received next period.
    IncomeNow : float
        Dollars in income received this period.
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
    hLvlGrid : np.array
        Exogenous grid of beginning-of-period health levels for use in the JDfix step.
    MedShkCount : int
        Number of non-zero medical need shocks to use in EGM step.
    HealthProdFunc : function
        Additional health produced as a function of hLvl and iLvl.
    MargHealthProdInvFunc : function
        Inverse of marginal health produced function.  Takes a marginal health produced
        and an hLvl, returns an iLvl with that marginal productivity.
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
    ConvexityFixer : JDfixer
        Instance of JDfixer that transforms irregular data grids onto exog grids.
        
    Returns
    -------
    solution_now : HealthInvSolution
        Solution to this period's health investment problem.
    '''
    # Define utility functions
    u = lambda C : CRRAutility(C,gam=CRRA)
    uP = lambda C : CRRAutilityP(C,gam=CRRA)
    uinv = lambda C : CRRAutility_inv(C,gam=CRRA)
    uPinv = lambda C : CRRAutilityP_inv(C,gam=CRRA)
    BequestMotive = lambda a : Bequest1*CRRAutility(a + Bequest0,gam=CRRA)
    BequestMotiveP = lambda a : Bequest1*CRRAutilityP(a + Bequest0,gam=CRRA)

    # Unpack next period's solution
    vFuncNext = solution_next.vFunc
    dvdbFuncNext = solution_next.dvdbFunc
#    dvdhFuncNext = solution_next.dvdhFunc
    if hasattr(vFuncNext,'vLim'):
        vLimNext = vFuncNext.vLim
    else: # This only happens in terminal period, when vFuncNext is a constant function
        vLimNext = 0.0
    vLimNow = DiscFac*vLimNext + LifeUtility
    
    # Make arrays of end-of-period assets and post-investment health
    aCount = aXtraGrid.size
    Hcount = Hgrid.size
    aLvlGrid = IncomeNow*aXtraGrid
    Harray = np.tile(np.reshape(Hgrid,(1,Hcount)),(aCount,1))
    
    # Make arrays of states we could arrive in next period
    hNextCount = 51
    hNextGrid = np.linspace(0.,1.,hNextCount)
    bNextGrid = Rfree*aLvlGrid + IncomeNext
    bNextArray = np.tile(np.reshape(bNextGrid,(aCount,1,1)),(1,1,hNextCount))
    hNextArray = np.tile(np.reshape(hNextGrid,(1,1,hNextCount)),(aCount,1,1))
    
    # Evaluate (marginal) value at the grid of future states, then tile
    vNext = vFuncNext(bNextArray,hNextArray)
    dvdbNext = dvdbFuncNext(bNextArray,hNextArray)
#    dvdhNext = dvdhFuncNext(bNextArray,hNextArray)
#    vNext[:,0,0] = BequestMotive(aLvlGrid) # Bequest motive when dead
#    dvdbNext[:,0,0] = BequestMotiveP(aLvlGrid) # Marginal bequest motive when dead
#    dvdhNext[:,0,0] = 0.0 # No value of additional health if dead
#    dvdhNext[:,0,-1] = 0.0 # No value of additional health if capped at 1
    vNext_tiled = np.tile(vNext,(1,Hcount,1))
    dvdbNext_tiled = np.tile(dvdbNext,(1,Hcount,1))
#    dvdhNext_tiled = np.tile(dvdhNext,(1,Hcount,1))
    BequestMotiveArray = np.tile(np.reshape(BequestMotive(aLvlGrid),(aCount,1)),(1,Hcount))
    BequestMotiveParray = np.tile(np.reshape(BequestMotiveP(aLvlGrid),(aCount,1)),(1,Hcount))
    
    # Calculate the probability of arriving at each future health state from each current health state
    Harray_temp = np.tile(np.reshape(Hgrid,(1,Hcount,1)),(1,1,hNextCount))
    hNextArray_temp = np.tile(np.reshape(hNextGrid,(1,1,hNextCount)),(1,Hcount,1))
    HealthShkStd = HealthShkStd0 + HealthShkStd1*Harray_temp
    zArray = (hNextArray_temp - Harray_temp)/HealthShkStd
    ProbArray = np.zeros((1,Hcount,hNextCount))
    BaseProbs = norm.pdf(zArray[:,:,1:-1]) # Don't include h=0 or h=1
    ProbSum = np.tile(np.sum(BaseProbs,axis=2,keepdims=True),(1,1,hNextCount-2))
    LivPrb = norm.sf(LivPrbFunc(Hgrid))
    TerriblePrb = norm.cdf(zArray[:,:,0])*LivPrb
    PerfectPrb = norm.sf(zArray[:,:,-1])*LivPrb
    BaseProbsAdj = BaseProbs/ProbSum*np.tile(np.reshape((LivPrb-TerriblePrb-PerfectPrb),(1,Hcount,1)),(1,1,hNextCount-2))
    ProbArray[:,:,1:-1] = BaseProbsAdj
    ProbArray[:,:,0] = TerriblePrb
    ProbArray[:,:,-1] = PerfectPrb
    DiePrb = 1. - LivPrb
        
    # Calculate the rate of change in probabilities of arriving in each future health state from end-of-period health
    H_eps = 0.0001
    Harray_temp = np.tile(np.reshape(Hgrid+H_eps,(1,Hcount,1)),(1,1,hNextCount))
    HealthShkStd = HealthShkStd0 + HealthShkStd1*Harray_temp
    zArray = (hNextArray_temp - Harray_temp)/HealthShkStd
    ProbArrayAlt = np.zeros((1,Hcount,hNextCount))
    BaseProbs = norm.pdf(zArray[:,:,1:-1]) # Don't include h=0 or h=1
    ProbSum = np.tile(np.sum(BaseProbs,axis=2,keepdims=True),(1,1,hNextCount-2))
    LivPrb = norm.sf(LivPrbFunc(Hgrid+H_eps))
    TerriblePrb = norm.cdf(zArray[:,:,0])*LivPrb
    PerfectPrb = norm.sf(zArray[:,:,-1])*LivPrb
    BaseProbsAdj = BaseProbs/ProbSum*np.tile(np.reshape((LivPrb-TerriblePrb-PerfectPrb),(1,Hcount,1)),(1,1,hNextCount-2))
    ProbArrayAlt[:,:,1:-1] = BaseProbsAdj
    ProbArrayAlt[:,:,0] = TerriblePrb
    ProbArrayAlt[:,:,-1] = PerfectPrb
    dProbdHArray = (ProbArrayAlt - ProbArray)/H_eps
    DiePrbAlt = 1. - LivPrb
    dDiePrbdH = (DiePrbAlt - DiePrb)/H_eps # This can actually be calculated in closed form: -norm.pdf(LivPrb)
    
    # Tile the probability arrays
    ProbArray_tiled = np.tile(ProbArray,(aCount,1,1))
    dProbdHArray_tiled = np.tile(dProbdHArray,(aCount,1,1))
    DiePrb_tiled = np.tile(np.reshape(DiePrb,(1,Hcount)),(aCount,1))
    dDiePrbdH_tiled = np.tile(np.reshape(dDiePrbdH,(1,Hcount)),(aCount,1))
    
    # Take expectations over future (marginal) value
    EndOfPrdv = DiscFac*np.sum(vNext_tiled*ProbArray_tiled,axis=2) + DiePrb_tiled*BequestMotiveArray
    EndOfPrddvda = Rfree*DiscFac*np.sum(dvdbNext_tiled*ProbArray_tiled,axis=2) + DiePrb_tiled*BequestMotiveParray
    EndOfPrddvdH = DiscFac*(np.sum(vNext_tiled*dProbdHArray_tiled,axis=2)) + dDiePrbdH_tiled*BequestMotiveArray
    MargValueRatio = EndOfPrddvda/EndOfPrddvdH
    MargValueRatioAdj = np.maximum(MargValueRatio*MedPrice,0.0)
    
    # Use a fixed point loop to find optimal health investment (unconstrained)
    tol = 0.00001
    LoopCount = 0
    MaxLoops = 20
    diff = np.ones_like(MargValueRatio)
    these = diff > tol
    points_left = np.sum(these)
    hNow = copy(Harray)
    iNow = np.zeros_like(Harray)
    while (points_left > 0) and (LoopCount < MaxLoops):
        Ratio = MargValueRatioAdj[these]
        H = Harray[these]
        hGuess = hNow[these].flatten()
        CopayGuess = CopayFunc(hGuess)
        iGuess = MargHealthProdInvFunc(CopayGuess*Ratio,hGuess)
        iGuess[np.isinf(Ratio)] = 0.0
        iGuess[Ratio == 0.] = 0.0
        hGuessNew = ExpHealthNextInvFunc(H - HealthProdFunc(iGuess,hGuess))
        diff[these] = np.abs(hGuess - hGuessNew)
        hNow[these] = hGuessNew
        iNow[these] = iGuess
        these = diff > tol
        points_left = np.sum(these)
        LoopCount += 1
        
    # Make a grid of medical need values
    LogMedShkMin = MedShkMeanFunc(1.0) - 3.0*MedShkStdFunc(1.0)
    LogMedShkMax = MedShkMeanFunc(0.0) + 5.0*MedShkStdFunc(0.0)
    #print(LogMedShkMin,LogMedShkMax)
    LogMedShkGrid = np.linspace(LogMedShkMin,LogMedShkMax,MedShkCount)
    MedShkGrid = np.insert(np.exp(LogMedShkGrid),0,0.0)
    ShkCount = MedShkGrid.size
    LogMedShkGridDense = np.linspace(LogMedShkMin,LogMedShkMax,MedShkCount*2)
    ShkGridDense = np.insert(np.exp(LogMedShkGridDense),0,0.0)
    
    # Make 3D arrays of states, health investment, insurance terms, and (marginal) values
    MedShkArrayBig = np.tile(np.reshape(MedShkGrid,(1,1,ShkCount)),(aCount,Hcount,1))
    aLvlArrayBig = np.tile(np.reshape(aLvlGrid,(aCount,1,1)),(1,Hcount,ShkCount))
    hLvlArrayBig = np.tile(np.reshape(hNow,(aCount,Hcount,1)),(1,1,ShkCount))
    iLvlArrayBig = np.tile(np.reshape(iNow,(aCount,Hcount,1)),(1,1,ShkCount))
    PremiumArrayBig = PremiumFunc(hLvlArrayBig)
    CopayArrayBig = CopayFunc(hLvlArrayBig)
    EndOfPrdvBig = np.tile(np.reshape(EndOfPrdv,(aCount,Hcount,1)),(1,1,ShkCount))
    EndOfPrddvdaBig = np.tile(np.reshape(EndOfPrddvda,(aCount,Hcount,1)),(1,1,ShkCount))
    EndOfPrddvdHBig = np.tile(np.reshape(EndOfPrddvdH,(aCount,Hcount,1)),(1,1,ShkCount))
    MedShkArrayAdj = MedShkArrayBig*MedPrice*CopayArrayBig
      
    # Use the first order conditions to calculate optimal expenditure on consumption and mitigative care (unconstrained)
    EndOfPrddvdaNvrs = uPinv(EndOfPrddvdaBig)
    cEffArrayBig = Gfunc(EndOfPrddvdaNvrs,MedShkArrayAdj)
    ShkZero = MedShkArrayBig == 0.
    cEffArrayBig[ShkZero] = EndOfPrddvdaNvrs[ShkZero]
    xLvlArrayBig = Dfunc(cEffArrayBig,MedShkArrayAdj)
    xLvlArrayBig[ShkZero] = cEffArrayBig[ShkZero]
    bLvlArrayBig = aLvlArrayBig + xLvlArrayBig + MedPrice*CopayArrayBig*iLvlArrayBig + PremiumArrayBig
    vArrayBig = u(cEffArrayBig) + LifeUtility + EndOfPrdvBig
    dvdhArrayBig = ExpHealthNextFunc.der(hLvlArrayBig)*EndOfPrddvdHBig
    
#    for j in range(ShkCount):
#        plt.plot(bLvlArrayBig[:,10,j],xLvlArrayBig[:,10,j])
#    plt.xlim(0.,1600.)
#    plt.ylim(0.,1600.)
#    plt.show()
#    
#    for j in range(ShkCount):
#        plt.plot(bLvlArrayBig[:,10,j],cEffArrayBig[:,10,j])
#    plt.xlim(0.,3000.)
#    plt.ylim(0.,3000.)
#    plt.show()
    
#    print('dvda',np.sum(np.isnan(EndOfPrddvdaBig)),np.sum(np.isinf(EndOfPrddvdaBig)))
#    print('MedShk',np.sum(np.isnan(MedShkArrayAdj)),np.sum(np.isinf(MedShkArrayAdj)))
#    print('iLvl',np.sum(np.isnan(iLvlArrayBig)),np.sum(np.isinf(iLvlArrayBig)))
#    print('xLvl',np.sum(np.isnan(xLvlArrayBig)),np.sum(np.isinf(xLvlArrayBig)))
#    print('bLvl',np.sum(np.isnan(bLvlArrayBig)),np.sum(np.isinf(bLvlArrayBig)))
    
    # Make an exogenous grid of bLvl and MedShk values where individual is constrained
    bCnstCount = 16
    PremiumTemp = PremiumFunc(hLvlArrayBig[0,:,:]) # Decent candidate for lower bound of bLvl
    MedShkArrayCnst = np.tile(np.reshape(MedShkGrid,(1,1,ShkCount)),(bCnstCount,Hcount,1))
    FractionGrid = np.tile(np.reshape(np.arange(1,bCnstCount+1,dtype=float)/(bCnstCount+1),(bCnstCount,1,1)),(1,Hcount,ShkCount))
    bLvlArrayCnst = np.tile(np.reshape(bLvlArrayBig[0,:,:]-PremiumTemp,(1,Hcount,ShkCount)),(bCnstCount,1,1))*FractionGrid + np.tile(np.reshape(PremiumTemp,(1,Hcount,ShkCount)),(bCnstCount,1,1))
    HarrayCnst = np.tile(np.reshape(Hgrid,(1,Hcount,1)),(bCnstCount,1,ShkCount))
    EndOfPrddvdHCnst = np.tile(np.reshape(EndOfPrddvdH[0,:],(1,Hcount,1)),(bCnstCount,1,ShkCount))
    EndOfPrddvdHCnstAdj = np.maximum(EndOfPrddvdHCnst,0.0)
    
    # Use a fixed point loop to solve for the constrained solution at each constrained bLvl
    tol = 0.00001
    LoopCount = 0
    MaxLoops = 20
    diff = np.ones_like(HarrayCnst)
    these = diff > tol
    points_left = np.sum(these)
    cEffNow = np.tile(np.reshape(cEffArrayBig[0,:,:],(1,Hcount,ShkCount)),(bCnstCount,1,1))
    hNow = copy(HarrayCnst)
    iNow = np.zeros_like(hNow)
    EffPriceNow = MedPrice*CopayFunc(hNow)
    xLvlNow = np.zeros_like(iNow)
    while (points_left > 0) and (LoopCount < MaxLoops):
        cEff = cEffNow[these]
        bLvl = bLvlArrayCnst[these]
        MedShk = MedShkArrayCnst[these]
        H = HarrayCnst[these]
        hGuess = hNow[these]
        EndOfPrddvdHNow = EndOfPrddvdHCnstAdj[these]
        EffPrice = EffPriceNow[these]
        dvdC = uP(cEff)/DpFunc(cEff,MedShk*EffPrice)
        ImpliedMargHealthProd = dvdC*EffPrice/EndOfPrddvdHNow
        iGuess = MargHealthProdInvFunc(ImpliedMargHealthProd,hGuess)
        hGuessNew = ExpHealthNextInvFunc(H - HealthProdFunc(iGuess,hGuess))
        EffPrice = MedPrice*CopayFunc(hGuessNew)
        Premium = PremiumFunc(hGuessNew)
        xLvl = bLvl - Premium - iGuess*EffPrice
        cEff = cEffFunc(xLvl,MedShk*EffPrice)
        ShkZero = MedShk == 0.
        cEff[ShkZero] = xLvl[ShkZero]
        diff[these] = np.abs(hGuessNew - hGuess)
        hNow[these] = hGuessNew
        iNow[these] = iGuess
        xLvlNow[these] = xLvl
        cEffNow[these] = cEff
        EffPriceNow[these] = EffPrice
        these = diff > tol
        points_left = np.sum(these)
        LoopCount += 1
        
    # Rename the constrained arrays and calculate (marginal) values for them
    cEffArrayCnst = cEffNow
    xLvlArrayCnst = xLvlNow
    hLvlArrayCnst = hNow
    iLvlArrayCnst = iNow
    vArrayCnst = u(cEffArrayCnst) + LifeUtility + np.tile(np.reshape(EndOfPrdvBig[0,:,:],(1,Hcount,ShkCount)),(bCnstCount,1,1))
    dvdhArrayCnst = ExpHealthNextFunc.der(hLvlArrayCnst)*EndOfPrddvdHCnst
    
#    print('iCnst',np.sum(np.isnan(iLvlArrayCnst)),np.sum(np.isinf(iLvlArrayCnst)))
#    print('xCnst',np.sum(np.isnan(xLvlArrayCnst)),np.sum(np.isinf(xLvlArrayCnst)))
#    print('vCnst',np.sum(np.isnan(vArrayCnst)),np.sum(np.isinf(vArrayCnst)))
    
    # Combine the constrained and unconstrained solutions into unified arrays
    bLvlArrayAll = np.concatenate((bLvlArrayCnst,bLvlArrayBig),axis=0)
    hLvlArrayAll = np.concatenate((hLvlArrayCnst,hLvlArrayBig),axis=0)
    MedShkArrayAll = np.concatenate((MedShkArrayCnst,MedShkArrayBig),axis=0)
    xLvlArrayAll = np.concatenate((xLvlArrayCnst,xLvlArrayBig),axis=0)
    iLvlArrayAll = np.concatenate((iLvlArrayCnst,iLvlArrayBig),axis=0)
    vArrayAll = np.concatenate((vArrayCnst,vArrayBig),axis=0)
    dvdhArrayAll = np.concatenate((dvdhArrayCnst,dvdhArrayBig),axis=0)
    vNvrsArrayAll = uinv(vArrayAll - vLimNow)
    
    # Apply the Jorgensen-Druedahl convexity fix and construct expenditure and investment functions
#    t_start = clock()
    xLvlArray, iLvlArray, vNvrsArray, dvdhArray = ConvexityFixer(bLvlArrayAll,hLvlArrayAll,MedShkArrayAll,
                                        vNvrsArrayAll,dvdhArrayAll,xLvlArrayAll,iLvlArrayAll,bLvlGrid,hLvlGrid,ShkGridDense)
#    t_end = clock()
#    print('JD fix took ' + str(t_end-t_start) + ' seconds.')
    xFuncNow = TrilinearInterp(xLvlArray,bLvlGrid,hLvlGrid,ShkGridDense)
    iFuncNow = TrilinearInterp(iLvlArray,bLvlGrid,hLvlGrid,ShkGridDense)
    PolicyFuncNow = HealthInvestmentPolicyFunc(xFuncNow,iFuncNow,bFromxFunc,CopayFunc)
    
    # Find the critical shock where the consumption floor begins to bind
    bLvlCount = bLvlGrid.size
    hLvlCount = hLvlGrid.size
    ShkCount  = MedShkGrid.size
    bLvlArray_temp = np.tile(np.reshape(bLvlGrid,(bLvlCount,1)),(1,hLvlCount))
    hLvlArray_temp = np.tile(np.reshape(hLvlGrid,(1,hLvlCount)),(bLvlCount,1))
    #MedShkMax = np.tile(np.reshape(np.exp(MedShkMeanFunc(hLvlGrid) + 5.0*MedShkStdFunc(hLvlGrid)),(1,hLvlCount)),(bLvlCount,1))
    MedShkMax = np.exp(MedShkMeanFunc(0.0) + 5.0*MedShkStdFunc(0.0))*np.ones_like(bLvlArray_temp)
    CritShkArray = 1e-8*np.ones_like(bLvlArray_temp) # Current guess of critical shock for each (bLvl,hLvl)
    DiffArray = np.ones_like(bLvlArray_temp) # Relative change in crit shock guess this iteration
    Unresolved = np.ones_like(bLvlArray_temp,dtype=bool) # Indicator for which points are still unresolved
    UnresolvedCount = Unresolved.size # Number of points whose CritShk has not been found
    DiffTol = 1e-5 # Convergence tolerance for the search
    LoopCount = 0
    LoopMax = 30
    while (UnresolvedCount > 0) and (LoopCount < LoopMax): # Loop until all points have converged on CritShk
        CritShkPrev = CritShkArray[Unresolved]
        bLvl_temp = bLvlArray_temp[Unresolved]
        hLvl_temp = hLvlArray_temp[Unresolved]            
        xLvl_temp = np.minimum(xFuncNow(bLvl_temp,hLvl_temp,CritShkPrev),bLvl_temp)
        EffPrice_temp = MedPrice*CopayFunc(hLvl_temp)
        CritShkNew = np.minimum(CritShkFunc(xLvl_temp)/EffPrice_temp,MedShkMax[Unresolved])
        DiffNew = np.abs(CritShkNew/CritShkPrev - 1.)                
        DiffArray[Unresolved] = DiffNew
        CritShkArray[Unresolved] = CritShkNew
        Unresolved[Unresolved] = DiffNew > DiffTol
        UnresolvedCount = np.sum(Unresolved)
        LoopCount += 1
        
    # Choose medical need shock grids for integration
    LogCritShkArray = np.log(CritShkArray)
    
#    for j in range(hLvlGrid.size):
#        plt.plot(bLvlGrid,LogCritShkArray[:,j])
#    plt.xlim(0.,200.)
#    plt.show()
    
#    for j in range(ShkGridDense.size):
#        plt.plot(bLvlGrid,vNvrsArray[:,3,j])
#    plt.ylim(0.,25.)
#    plt.xlim(0.,25.)
#    plt.show()
    
#    for j in range(ShkGridDense.size):
#        plt.plot(bLvlGrid,xLvlArray[:,20,j])
#    plt.ylim(0.,1600.)
#    plt.show()
    
#    for j in range(hLvlGrid.size):
#        plt.plot(bLvlGrid,vNvrsArray[:,j,20])
#    plt.ylim(-10.,2000.)
#    plt.show()
    
    MedShkMeanArray = np.tile(np.reshape(MedShkMeanFunc(hLvlGrid),(1,hLvlCount,1)),(bLvlCount,1,MedShkCount))
    MedShkStdArray = np.tile(np.reshape(MedShkStdFunc(hLvlGrid),(1,hLvlCount,1)),(bLvlCount,1,MedShkCount))
#    DevArray = np.tile(np.reshape(np.linspace(0.,8.,MedShkCount),(1,1,MedShkCount)),(bLvlCount,hLvlCount,1))
#    LogMedShkArray = np.tile(np.reshape(LogCritShkArray,(bLvlCount,hLvlCount,1)),(1,1,MedShkCount)) - DevArray*MedShkStdArray
    LogMedShkLowerArray = MedShkMeanArray - 3.0*MedShkStdArray
    FracArray = np.tile(np.reshape(np.linspace(0.0,0.95,MedShkCount),(1,1,MedShkCount)),(bLvlCount,hLvlCount,1))
    LogMedShkArray = LogMedShkLowerArray + FracArray*(np.tile(np.reshape(LogCritShkArray,(bLvlCount,hLvlCount,1)),(1,1,MedShkCount)) - LogMedShkLowerArray)
    MedShkValArray = np.exp(LogMedShkArray)
    
    # Calculate probabilities of all of the medical shocks
    zArray = (LogMedShkArray - MedShkMeanArray)/MedShkStdArray
    BasePrbArray = norm.pdf(zArray)
    CritShkPrbArray = norm.sf((LogCritShkArray - MedShkMeanArray[:,:,0])/MedShkStdArray[:,:,0])
    SumPrbArray = np.sum(BasePrbArray,axis=2)
    AdjArray = np.tile(np.reshape((1.0-CritShkPrbArray)/SumPrbArray,(bLvlCount,hLvlCount,1)),(1,1,MedShkCount))
    MedShkPrbArray = BasePrbArray*AdjArray
    
    # Calculate the change in probabilities of the medical shocks as h increases slightly
    h_eps = 0.0001
    MedShkMeanArray = np.tile(np.reshape(MedShkMeanFunc(hLvlGrid+h_eps),(1,hLvlCount,1)),(bLvlCount,1,MedShkCount))
    MedShkStdArray = np.tile(np.reshape(MedShkStdFunc(hLvlGrid+h_eps),(1,hLvlCount,1)),(bLvlCount,1,MedShkCount))
    zArray = (LogMedShkArray - MedShkMeanArray)/MedShkStdArray
    BasePrbArray = norm.pdf(zArray)
    CritShkPrbArrayAlt = norm.sf((LogCritShkArray - MedShkMeanArray[:,:,0])/MedShkStdArray[:,:,0])
    SumPrbArray = np.sum(BasePrbArray,axis=2)
    AdjArray = np.tile(np.reshape((1.0-CritShkPrbArrayAlt)/SumPrbArray,(bLvlCount,hLvlCount,1)),(1,1,MedShkCount))
    MedShkPrbArrayAlt = BasePrbArray*AdjArray
    dMedShkPrbdhArray = (MedShkPrbArrayAlt - MedShkPrbArray)/h_eps
    dCritShkPrbdhArray = (CritShkPrbArrayAlt - CritShkPrbArray)/h_eps
    
    # Make an array of values that are attained if we hit the Cfloor this period
    EndOfPrdvFunc_no_assets = ValueFunc(LinearInterp(Hgrid,uinv(EndOfPrdv[0,:]-DiscFac*vLimNext)),CRRA)
    EndOfPrddvdhFunc_no_assets = LinearInterp(Hgrid,EndOfPrddvdH[0,:])
    Hgrid_temp = ExpHealthNextFunc(hLvlGrid)
    dHdh_temp = ExpHealthNextFunc.der(hLvlGrid)
    vFloorArray = np.tile(np.reshape(u(Cfloor) + LifeUtility + EndOfPrdvFunc_no_assets(Hgrid_temp),(1,hLvlCount)),(bLvlCount,1)) + DiscFac*vLimNext
    dvdhFloorArray = np.tile(np.reshape(dHdh_temp*EndOfPrddvdhFunc_no_assets(Hgrid_temp),(1,hLvlCount)),(bLvlCount,1))
    
    # Find where each shock for integration falls on the MedShkGridDense
    IdxHi = np.minimum(np.searchsorted(ShkGridDense,MedShkValArray),ShkGridDense.size-1)
    IdxLo = IdxHi - 1
    ShkLo = ShkGridDense[IdxLo]
    ShkHi = ShkGridDense[IdxHi]
    alpha = (MedShkValArray - ShkLo)/(ShkHi - ShkLo)
    alpha_comp = 1.0 - alpha
    bIdx  = np.tile(np.reshape(np.arange(bLvlCount),(bLvlCount,1,1)),(1,hLvlCount,MedShkCount))
    hIdx  = np.tile(np.reshape(np.arange(hLvlCount),(1,hLvlCount,1)),(bLvlCount,1,MedShkCount))
    
    # Integrate value according to the shock probabilities
    vNvrs_temp = alpha_comp*vNvrsArray[bIdx,hIdx,IdxLo] + alpha*vNvrsArray[bIdx,hIdx,IdxHi]
    v_temp = u(vNvrs_temp) + vLimNow
    ValueArrayFlat = np.sum(v_temp*MedShkPrbArray,axis=2) + CritShkPrbArray*vFloorArray
    vNvrsArrayFlat = uinv(ValueArrayFlat - vLimNow)
    
    # Integrate marginal value of bank balances according to the shock probabilities
    x_temp = alpha_comp*xLvlArray[bIdx,hIdx,IdxLo] + alpha*xLvlArray[bIdx,hIdx,IdxHi]
    Copay_temp = np.tile(np.reshape(CopayFunc(hLvlGrid),(1,hLvlCount,1)),(bLvlCount,1,MedShkCount))*MedPrice
    cEff_temp = cEffFunc(x_temp,MedShkValArray*Copay_temp)
    Dp_temp = DpFunc(x_temp,MedShkValArray*Copay_temp)
    dvdb_temp = uP(cEff_temp)/Dp_temp
    dvdbArrayFlat = np.sum(dvdb_temp*MedShkPrbArray,axis=2)
    dvdbNvrsArrayFlat = uPinv(dvdbArrayFlat)
    
    # Integrate marginal value of health according to the shock probabilities
    dvdh_temp = alpha_comp*dvdhArray[bIdx,hIdx,IdxLo] + alpha*dvdhArray[bIdx,hIdx,IdxHi]
    dvdhArrayFlat = dCritShkPrbdhArray*vFloorArray + np.sum(v_temp*dMedShkPrbdhArray,axis=2) + np.sum(dvdh_temp*MedShkPrbArray,axis=2) + CritShkPrbArray*dvdhFloorArray
    
    # Make (marginal) value functions
    vNvrsFuncNow = BilinearInterp(vNvrsArrayFlat,bLvlGrid,hLvlGrid)
    vFuncNow = ValueFunc2D(vNvrsFuncNow,CRRA,vLimNow)
    dvdbNvrsFuncNow = BilinearInterp(dvdbNvrsArrayFlat,bLvlGrid,hLvlGrid)
    dvdbFuncNow = MargValueFunc2D(dvdbNvrsFuncNow,CRRA)
    dvdhFuncNow = BilinearInterp(dvdhArrayFlat,bLvlGrid,hLvlGrid)
    
#    # Make an alternate dvdhFunc using first differences of vFunc
#    h_eps = 0.0001
#    b_temp = np.tile(np.reshape(bLvlGrid,(bLvlGrid.size,1)),(1,hLvlGrid.size))
#    h_temp = np.tile(np.reshape(hLvlGrid,(1,hLvlGrid.size)),(bLvlGrid.size,1))
#    dvdhArrayAlt = (vFuncNow(b_temp,h_temp+h_eps) - vFuncNow(b_temp,h_temp))/h_eps
#    dvdhFuncNowAlt = BilinearInterp(dvdhArrayAlt,bLvlGrid,hLvlGrid)
    
    # Package and return the solution object
    solution_now = HealthInvestmentSolution(PolicyFuncNow,vFuncNow,dvdbFuncNow,dvdhFuncNow)
#    solution_now.dvdhFuncAlt = dvdhFuncNowAlt
    return solution_now
    
    

class HealthInvestmentConsumerType(IndShockConsumerType):
    '''
    A class for representing agents in the health investment model.
    '''
    
    def __init__(self,**kwds):
        AgentType.__init__(self,solution_terminal=None,time_flow=True,pseudo_terminal=True,**kwds)
        self.time_inv = ['CRRA','DiscFac','MedCurve','Cfloor','LifeUtility','MargUtilityShift',
                         'Rfree','Bequest0','Bequest1','MedShkCount','HealthProd0','HealthProd1',
                         'HealthProd2','HealthShkStd0','HealthShkStd1']
        self.time_vary = []
        self.poststate_vars = ['aLvlNow','HlvlNow']
        self.solveOnePeriod = solveHealthInvestment
    
    
    def updateMedShkDstnFuncs(self):
        '''
        Constructs the attributes MedShkMeanFunc as a time-varying attribute and
        MedShkStdFunc as a time-invarying attribute.
        Each element of these lists is a real to real function that takes in a
        health level and returns a mean or stdev of log medical need.
        
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
            Age = t*1
            beta0 = self.MedShkMean0 + self.Sex*self.MedShkMeanSex + self.MedShkMeanAge*Age + self.MedShkMeanAgeSq*Age**2
            beta1 = self.MedShkMeanHealth
            beta2 = self.MedShkMeanHealthSq
            MedShkMeanFunc.append(QuadraticFunction(beta0,beta1,beta2))
        
        self.MedShkMeanFunc = MedShkMeanFunc
        self.MedShkStdFunc = MedShkStdFunc
        self.addToTimeVary('MedShkMeanFunc')
        self.addToTimeInv('MedShkStdFunc')
        
        if not orig_time:
            self.timeRev()
        
            
    def updateHealthTransFuncs(self):
        '''
        Constructs the attributes LivPrbFunc, ExpHealthNextFunc, and ExpHealthNextInvFunc
        as time-varying lists.  Each element of these lists is a real to real function
        that takes in a health level and returns a survival probability, expected next
        period health, or (inverse) this period health.
        
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
            
        LivPrbFunc = []
        ExpHealthNextFunc = []
        ExpHealthNextInvFunc = []
        for t in range(self.T_cycle):
            Age = t
            theta0 = self.Mortality0 + self.Sex*self.MortalitySex + self.MortalityAge*Age + self.MortalityAgeSq*Age**2
            theta1 = self.MortalityHealth
            theta2 = self.MortalityHealthSq
            LivPrbFunc.append(QuadraticFunction(theta0,theta1,theta2))
            
            gamma0 = self.HealthNext0 + self.Sex*self.HealthNextSex + self.HealthNextAge*Age + self.HealthNextAgeSq*Age**2
            gamma1 = self.HealthNextHealth
            gamma2 = self.HealthNextHealthSq
            ThisHealthFunc = QuadraticFunction(gamma0,gamma1,gamma2)
            ExpHealthNextFunc.append(ThisHealthFunc)
            ExpHealthNextInvFunc.append(ThisHealthFunc.inverse)
        LivPrbFunc.pop() # Replace last period's LivPrb with (effectively) zero
        LivPrbFunc.append(QuadraticFunction(20.0,0.0,0.0))
        
        self.LivPrbFunc = LivPrbFunc
        self.ExpHealthNextFunc = ExpHealthNextFunc
        self.ExpHealthNextInvFunc = ExpHealthNextInvFunc
        self.addToTimeVary('LivPrbFunc','ExpHealthNextFunc','ExpHealthNextInvFunc')
        
        if not orig_time:
            self.timeRev()
            
            
    def updateInsuranceFuncs(self):
        '''
        Constructs the attributes PremiumFunc and CopayFunc as time-varying lists
        of real to real functions.  Each element of these lists takes in a current
        health level and returns a coinsurance rate or premium.
        
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
            
        PremiumFunc = []
        CopayFunc = []
        for t in range(self.T_cycle):
            Age = t*1
            y = self.IncomeNow[t]
            p0 = self.Premium0 + self.Sex*self.PremiumSex + self.PremiumAge*Age + self.PremiumAgeSq*Age**2 + self.PremiumInc*y + self.PremiumIncSq*y**2 + self.PremiumIncCu*y**3
            p1 = self.PremiumHealth + self.PremiumHealthAge*Age + self.PremiumHealthAgeSq*Age**2 + self.PremiumHealthInc*y + self.PremiumHealthIncSq*y**2
            p2 = self.PremiumHealthSq + self.PremiumHealthSqAge*Age + self.PremiumHealthSqAgeSq*Age**2 + self.PremiumHealthSqInc*y + self.PremiumHealthSqIncSq*y**2
            PremiumFunc.append(QuadraticFunction(p0,p1,p2))
            
            c0 = self.Copay0 + self.Sex*self.CopaySex + self.CopayAge*Age + self.CopayAgeSq*Age**2 + self.CopayInc*y + self.CopayIncSq*y**2 + self.CopayIncCu*y**3
            c1 = self.CopayHealth + self.CopayHealthAge*Age + self.CopayHealthAgeSq*Age**2 + self.CopayHealthInc*y + self.CopayHealthIncSq*y**2
            c2 = self.CopayHealthSq + self.CopayHealthSqAge*Age + self.CopayHealthSqAgeSq*Age**2 + self.CopayHealthSqInc*y + self.CopayHealthSqIncSq*y**2
            CopayFunc.append(QuadraticFunction(c0,c1,c2))
            
        self.PremiumFunc = PremiumFunc
        self.CopayFunc = CopayFunc
        self.addToTimeVary('PremiumFunc','CopayFunc')
            
        if not orig_time:
            self.timeRev()
            
            
    def updateStateGrids(self):
        '''
        Constructs the attributes aXtraGrid, bNrmGrid, Hgrid, and hGrid.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        None
        '''
        self.updateAssetsGrid()
        bNrmGrid = makeGridExpMult(ming=self.aXtraMin, maxg=self.aXtraMax, ng=self.bNrmCount, timestonest=self.aXtraNestFac)
        bNrmGrid = np.insert(bNrmGrid,0,0.0)
        self.aXtraGrid = np.insert(self.aXtraGrid,0,0.0)
        
        hGrid = np.linspace(0.,1.,self.hCount)
        Hgrid = np.linspace(-0.1,1.1,self.Hcount)
        
        self.bNrmGrid = bNrmGrid
        self.hLvlGrid = hGrid
        self.Hgrid = Hgrid
        
        bLvlGrid = []
        for t in range(self.T_cycle):
            bLvlGrid.append(bNrmGrid*self.IncomeNow[t] + self.IncomeNow[t])
        self.bLvlGrid = bLvlGrid
        
        self.addToTimeInv('bNrmGrid','hLvlGrid','Hgrid')
        self.addToTimeVary('bLvlGrid')

        
    def makeQuarticIncomePath(self):
        '''
        Constructs the time-varying attribute called income using a fourth degree
        polynomial for income by age.
        
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
            
        Age = np.arange(self.T_cycle)
        IncomeVec = self.Income0 + self.IncomeAge*Age + self.IncomeAgeSq*Age**2 + self.IncomeAgeCu*Age**3 + self.IncomeAgeQu*Age**4
        IncomeNow = IncomeVec.tolist()
        IncomeNext = (np.append(IncomeVec[1:],1.0)).tolist() # Income in period "after" terminal period is irrelevant
        
        if not orig_time:
            IncomeNow.reverse()
            IncomeNext.reverse
        self.IncomeNow = IncomeNow
        self.IncomeNext = IncomeNext
        self.addToTimeVary('IncomeNow','IncomeNext')
        
        if not orig_time:
            self.timeRev()
        
        
    def updateFirstOrderConditionFuncs(self):
        '''
        Constructs the time-invariant attributes bFromxFunc, Dfunc, DpFunc, Gfunc,
        cEffFunc, and CritShkFunc.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        None
        '''
        xLvl_aug_factor = 6
        MedShk_aug_factor = 6
        if self.time_flow:
            t0 = 0
            T = -1
        else:
            t0 = -1
            T = 0
        
        # Make a master grid of expenditure levels
        IncomeAdj = np.max(np.array(self.IncomeNow))
        xLvlGrid = makeGridExpMult(IncomeAdj*self.aXtraMin,IncomeAdj*self.aXtraMax,self.aXtraCount*xLvl_aug_factor,self.aXtraNestFac)
        
        # Make a master grid of medical need shocks
        ShkLogMin = self.MedShkMeanFunc[t0](1.0) - 3.0*self.MedShkStdFunc(1.0)
        ShkLogMax = self.MedShkMeanFunc[T](0.0) + 5.0*self.MedShkStdFunc(0.0)
        MedShkGrid = np.insert(np.exp(np.linspace(ShkLogMin,ShkLogMax,num=MedShk_aug_factor*self.MedShkCount)),0,0.0)
        
        # Make the bFromxFunc for this type
        bFromxFunc = makebFromxFunc(xLvlGrid,MedShkGrid,self.CRRA,self.MedCurve)
        
        # Unpack basics from the bFromxFunc
        bGrid = bFromxFunc.f_values
        xVec = bFromxFunc.x_list
        ShkVec = bFromxFunc.y_list
        
        # Construct a grid of effective consumption values
        xGrid = np.tile(np.reshape(xVec,(xVec.size,1)),(1,ShkVec.size))
        ShkGrid = np.tile(np.reshape(ShkVec,(1,ShkVec.size)),(xVec.size,1))
        ShkZero = ShkGrid == 0.
        qGrid = np.exp(-bGrid)
        cGrid = 1./(1.+qGrid)*xGrid
        MedGrid = qGrid/(1.+qGrid)*xGrid
        cEffGrid = (1.-np.exp(-MedGrid/ShkGrid))**self.MedCurve*cGrid
        cEffGrid[:,0] = xVec

        # Construct a grid of marginal effective consumption values
        temp1 = np.exp(-MedGrid/ShkGrid)
        temp2 = np.exp(MedGrid/ShkGrid)
        dcdx = temp2/(temp2 + self.MedCurve)
        dcdx[ShkZero] = 1.
        dMeddx = self.MedCurve/(temp2 + self.MedCurve)
        dMeddx[ShkZero] = 0.
        dcEffdx = self.MedCurve/ShkGrid*temp1*dMeddx*(1.-temp1)**(self.MedCurve-1.)*cGrid + (1.-temp1)**self.MedCurve*dcdx
        dcEffdx[ShkZero] = 1.
        dDdcEff = (dcEffdx)**(-1.) # Inverting the derivative
        dDdcEff[0,:] = 2*dDdcEff[1,:] - dDdcEff[2,:]
        
        # Calculate the grid to be used to make the G functions
        gothicvPnvrs = cEffGrid*dDdcEff**(1./self.CRRA)

        # Construct a 2D interpolation for the Dfunc, DpFunc, and Gfunc
        DfuncBase_by_Shk_list = []
        DpFuncBase_by_Shk_list = []
        GfuncBase_by_Shk_list = []
        for j in range(ShkVec.size):
            DfuncBase_by_Shk_list.append(LinearInterp(np.log(cEffGrid[1:,j]),np.log(xVec[1:]),lower_extrap=True))
            temp = -np.log(dDdcEff[1:,j]-1.0)
            DpFuncBase_by_Shk_list.append(LinearInterp(np.log(cEffGrid[1:,j]),temp,lower_extrap=True))
            GfuncBase_by_Shk_list.append(LinearInterp(np.log(gothicvPnvrs[1:,j]),np.log(cEffGrid[1:,j]),lower_extrap=True))
        Dfunc = LogOnLogFunc2D(LinearInterpOnInterp1D(DfuncBase_by_Shk_list,ShkVec))
        DpFunc = MargCostFunc(LinearInterpOnInterp1D(DpFuncBase_by_Shk_list,ShkVec))
        Gfunc = LogOnLogFunc2D(LinearInterpOnInterp1D(GfuncBase_by_Shk_list,ShkVec))
        
        # Make a 2D interpolation for the cEffFunc
        LogcEffGrid = np.log(cEffGrid[1:,:])
        LogxVec = np.log(xVec[1:])
        cEffFunc = LogOnLogFunc2D(BilinearInterp(LogcEffGrid,LogxVec,ShkVec))
        cEffFunc = cEffFunc
        
        # For each log(x) value, find the critical shock where the consumption floor binds
        N = LogcEffGrid.shape[0]
        LogCfloor = np.log(self.Cfloor)
        IdxVec = np.maximum(np.minimum(np.sum(LogcEffGrid >= LogCfloor,axis=1),ShkVec.size-1),1)
        LogC1 = LogcEffGrid[np.arange(N),IdxVec]
        LogC0 = LogcEffGrid[np.arange(N),IdxVec-1]
        alpha = (LogCfloor-LogC0)/(LogC1-LogC0)
        CritShk = np.maximum((1.-alpha)*ShkVec[IdxVec-1] + alpha*ShkVec[IdxVec],0.)
        i = np.sum(CritShk==0.)
        CritShkFunc = LinearInterp(np.insert(xVec[1:],i,self.Cfloor),np.insert(CritShk,i,0.0),lower_extrap=True)
        
        # Save the constructed functions as attributes of self
        self.bFromxFunc = bFromxFunc
        self.Dfunc = Dfunc
        self.DpFunc = DpFunc
        self.Gfunc = Gfunc
        self.cEffFunc = cEffFunc
        self.CritShkFunc = CritShkFunc
        self.addToTimeInv('bFromxFunc','Dfunc','DpFunc','Gfunc','cEffFunc','CritShkFunc')
        
        
    def updateHealthProdFuncs(self):
        '''
        Defines the time-invariant attributes HealthProdFunc, HealthProdInvFunc,
        MargHealthProdFunc, and MargHealthProdInvFunc.
        '''
        # Define the (inverse) health production function
        self.HealthProdFunc = lambda i,h : (self.HealthProd1 + h*self.HealthProd2)*i**self.HealthProd0
        self.HealthProdInvFunc = lambda x,h : (x/(self.HealthProd1 + h*self.HealthProd2))**(1./self.HealthProd0)
        self.MargHealthProdFunc = lambda i,h : self.HealthProd0*(self.HealthProd1 + h*self.HealthProd2)*i**(self.HealthProd0-1.)
        self.MargHealthProdInvFunc = lambda x,h : (x/(self.HealthProd0*(self.HealthProd1 + h*self.HealthProd2)))**(1./(self.HealthProd0-1.))
        self.addToTimeInv('HealthProdFunc','HealthProdInvFunc','MargHealthProdFunc','MargHealthProdInvFunc')
        
        
    def updateConvexityFixer(self):
        '''
        Creates the time-invariant attribute ConvexityFixer as an instance of JDfix.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        None
        '''
        self.ConvexityFixer = JDfixer(self.aXtraGrid.size+16,self.Hgrid.size,self.MedShkCount+1,
                                      self.bNrmGrid.size,self.hLvlGrid.size,self.MedShkCount*2+1)
        self.addToTimeInv('ConvexityFixer')
        
        
    def updateSolutionTerminal(self):
        '''
        Makes the attribute solution_terminal with constant zero value and marginal
        value.  Sets the attribute pseudo-terminal to True so that this object is
        not included in the solution after the solve method is run.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        None
        '''
        self.solution_terminal = HealthInvestmentSolution(PolicyFunc=NullFunc(), vFunc=ConstantFunction(0.), dvdbFunc=ConstantFunction(0.), dvdhFunc=ConstantFunction(0.))
        self.solution_terminal.dvdhFuncAlt = ConstantFunction(0.)
        self.pseudo_terminal = True


    def makeConstantMedPrice(self):
        '''
        Dummy method to fill in MedPrice as a constant value at every age.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        None
        '''
        self.MedPrice = self.T_cycle*[self.MedPrice0]
        self.addToTimeVary('MedPrice')
        
        
    def takeVaryingMedPrice(self,MedPriceHistory,t0):
        '''
        Method to generate a time-varying sequence of MedPrice values based on an
        absolute time history of MedPrice and an initial period t0.
        
        Parameters
        ----------
        MedPriceHistory : np.array
            History of MedPrice over absolute time (not agent-age-time).
        t0 : int
            Period of absolute history when this instance is "born".
        
        Returns
        -------
        None
        '''
        orig_time = self.time_flow
        if not self.time_flow:
            self.timeFwd()
            
        t1 = t0 + self.T_cycle
        MedPriceArray = MedPriceHistory[t0:t1]
        self.MedPrice = MedPriceArray.tolist()
        self.addToTimeVary('MedPrice')
        
        if not orig_time:
            self.timeRev()
        
             
    def update(self):
        '''
        Calls all the various update methods to preconstruct objects for the solver.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        None
        '''
        self.updateStateGrids()
        self.updateInsuranceFuncs()
        self.updateMedShkDstnFuncs()
        self.updateHealthTransFuncs()
        self.updateHealthProdFuncs()
        self.updateSolutionTerminal()
        self.updateFirstOrderConditionFuncs()
        self.updateConvexityFixer()
        
        
    def initializeSim(self):
        '''
        Prepares for a new simulation run by clearing histories and post-state
        variable arrays, and setting time to zero.
        '''
        self.resetRNG()
        self.t_sim = 0
        self.t_age = 0
        self.t_cycle = np.zeros(self.AgentCount,dtype=int)
        blank_array = np.zeros(self.AgentCount)
        for var_name in self.poststate_vars:
            exec('self.' + var_name + ' = copy(blank_array)')
        self.clearHistory()
        self.ActiveNow = np.zeros(self.AgentCount,dtype=bool)
        self.DiedNow = np.zeros(self.AgentCount,dtype=bool)
        self.hLvlNow = np.zeros(self.AgentCount) + np.nan
        self.CumLivPrb = np.zeros(self.AgentCount)
        self.DiePrbNow = np.zeros(self.AgentCount)
        
        
    def getMortality(self):
        '''
        Overwrites the standard method in AgentType with a simple thing.
        '''
        self.simDeath()
        self.simBirth()


    def simBirth(self):
        '''
        Activates agents who enter the data in this period.
        '''
        t = self.t_sim
        activate = self.BornBoolArray[t,:]
        self.ActiveNow[activate] = True
        self.aLvlNow[activate] = self.aLvlInit[activate]
        self.HlvlNow[activate] = self.HlvlInit[activate]
        self.CumLivPrb[activate] = 1.0

    
#    def simDeath(self):
#        '''
#        Kills agents based on their probit mortality function.
#        '''
#        these = self.ActiveNow
#        t = self.t_sim
#        N = np.sum(these)
#        
#        MortShkNow = drawNormal(N,seed=self.RNG.randint(0,2**31-1))
#        if t > 0: # Draw on LivPrbFunc from *previous* age into this age
#            CritShk = self.LivPrbFunc[t-1](self.HlvlNow[these])
#        else: # Shouldn't be any agents yet, but just in case...
#            CritShk = np.ones(N) - np.inf
#        kill = MortShkNow < CritShk
#        just_died = np.zeros(self.AgentCount,dtype=bool)
#        just_died[these] = kill
#        
#        self.hLvlNow[just_died] = 0.0
#        self.ActiveNow[just_died] = False
#        self.DiedNow[just_died] = True

    def simDeath(self):
        '''
        Calculates death probability for each simulated agent and updates population
        weights for active agents.  Does not "kill" agents by removing them from sim.
        '''
        these = self.ActiveNow
        t = self.t_sim
        N = np.sum(these)
        
        # Get survival and mortality probabilities
        if t > 0: # Draw on LivPrbFunc from *previous* age into this age
            LivPrb = norm.sf(self.LivPrbFunc[t-1](self.HlvlNow[these]))
        else: # Shouldn't be any agents yet, but just in case...
            LivPrb = np.ones(N)
        DiePrb = 1. - LivPrb
        
        # Apply survival probabilities to active agents and store mortality probabilities
        self.CumLivPrb[these] *= LivPrb
        self.DiePrbNow[these] = DiePrb
    
    
    def getShocks(self):
        '''
        Draws health shocks and base values for the medical need shock.
        '''
        these = self.ActiveNow
        not_these = np.logical_not(these)
        N = np.sum(these)
        
        HlvlNow = self.HlvlNow[these]
        HealthShkStd = self.HealthShkStd0 + self.HealthShkStd1*HlvlNow
        hShkNow = drawNormal(N,seed=self.RNG.randint(0,2**31-1))*HealthShkStd
        if ~hasattr(self,'hShkNow'):
            self.hShkNow = np.zeros(self.AgentCount)
        self.hShkNow[these] = hShkNow
        self.hShkNow[not_these] = np.nan
        
        MedShkBase = drawNormal(N,seed=self.RNG.randint(0,2**31-1))
        if ~hasattr(self,'MedShkBase'):
            self.MedShkBase = np.zeros(self.AgentCount)
        self.MedShkBase[these] = MedShkBase
        self.MedShkBase[not_these] = np.nan

 
    def getStates(self):
        '''
        Calculates hLvlNow, bLvlNow, and MedShkNow using aLvlNow, HlvlNow,
        hShkNow, and MedShkBase.
        '''
        hLvlNow = np.maximum(np.minimum(self.HlvlNow + self.hShkNow,1.0),0.001)
        #just_died = hLvlNow == 0.
        #self.ActiveNow[just_died] = False
        #self.DiedNow[just_died] = True
        hLvlNow[self.HlvlNow == 0.] = np.nan
        self.hLvlNow = hLvlNow
        
        these = self.ActiveNow
        not_these = np.logical_not(these)
        t = self.t_sim
        
        MedShkMean = self.MedShkMeanFunc[t](hLvlNow)
        MedShkStd = self.MedShkStdFunc(hLvlNow)
        LogMedShkNow = MedShkMean + self.MedShkBase*MedShkStd
        MedShkNow = np.exp(LogMedShkNow)
        self.MedShkNow = MedShkNow
        
        bLvlNow = self.Rfree*self.aLvlNow + self.IncomeNow[t]
        bLvlNow[not_these] = np.nan
        self.bLvlNow = bLvlNow
        
        
    def getControls(self):
        '''
        Evaluates control variables cLvlNow, iLvlNow, MedLvlNow using state variables.
        '''
        t = self.t_sim
        these = self.ActiveNow
        not_these = np.logical_not(these)
        
        PremiumNow = self.PremiumFunc[t](self.hLvlNow[these])
        CopayNow = self.CopayFunc[t](self.hLvlNow[these])
        
        bLvlNow = self.bLvlNow[these]
        hLvlNow = self.hLvlNow[these]
        MedShkNow = self.MedShkNow[these]
        cLvlNow, MedLvlNow, iLvlNow, xLvlNow = self.solution[t].PolicyFunc(bLvlNow,hLvlNow,MedShkNow)
        iLvlNow = np.maximum(iLvlNow,0.)
        
        MedPriceEff = self.MedPrice[t]*CopayNow
        MedShkEff = MedShkNow*MedPriceEff
        cEffNow = (1. - np.exp(-MedLvlNow/(MedShkEff)))*cLvlNow
        BelowCfloor = cEffNow < self.Cfloor
        
        xLvlNow[BelowCfloor] = self.Dfunc(self.Cfloor*np.ones_like(MedShkEff[BelowCfloor]),MedShkEff[BelowCfloor])
        iLvlNow[BelowCfloor] = 0.0
        cShareTrans = self.bFromxFunc(xLvlNow[BelowCfloor],MedShkEff[BelowCfloor])
        q = np.exp(-cShareTrans)
        cLvlNow[BelowCfloor] = xLvlNow[BelowCfloor]/(1.+q)
        MedLvlNow[BelowCfloor] = xLvlNow[BelowCfloor]*q/(1.+q)
        
        if ~hasattr(self,'cLvlNow'):
            self.cLvlNow = np.zeros(self.AgentCount)
            self.MedLvlNow = np.zeros(self.AgentCount)
            self.iLvlNow = np.zeros(self.AgentCount)
            self.xLvlNow = np.zeros(self.AgentCount)
            self.PremiumNow = np.zeros(self.AgentCount)
            self.CopayNow = np.zeros(self.AgentCount)
            
        self.PremiumNow[these] = PremiumNow
        self.CopayNow[these] = CopayNow
        self.cLvlNow[these] = cLvlNow
        self.MedLvlNow[these] = MedLvlNow
        self.iLvlNow[these] = iLvlNow
        self.xLvlNow[these] = xLvlNow
        self.PremiumNow[not_these] = np.nan
        self.CopayNow[not_these] = np.nan
        self.cLvlNow[not_these] = np.nan
        self.MedLvlNow[not_these] = np.nan
        self.iLvlNow[not_these] = np.nan
        self.xLvlNow[not_these] = np.nan
        
        
    def getPostStates(self):
        '''
        Calculates post states aLvlNow and HlvlNow.
        '''
        t = self.t_sim
        aLvlNow = self.bLvlNow - self.PremiumNow - self.xLvlNow - self.CopayNow*self.MedPrice[t]*self.iLvlNow
        aLvlNow = np.maximum(aLvlNow,0.0) # Fixes those who go negative due to Cfloor help
        HlvlNow = self.ExpHealthNextFunc[t](self.hLvlNow) + self.HealthProdFunc(self.iLvlNow,self.hLvlNow)
        self.aLvlNow = aLvlNow
        self.HlvlNow = HlvlNow
        
        self.TotalMedNow = self.MedPrice[t]*(self.MedLvlNow + self.iLvlNow)
        self.OOPmedNow = self.TotalMedNow*self.CopayNow

    
    def plotxFuncByHealth(self,t,MedShk,bMin=None,bMax=20.0,hSet=None):
        '''
        Plot the expenditure function vs bLvl at a fixed medical need shock and
        a set of health values.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        None
        '''
        if hSet is None:
            hSet = self.solution[t].PolicyFunc.xFunc.y_list
        if bMin is None:
            bMin = self.solution[t].PolicyFunc.xFunc.x_list[0]
            
        B = np.linspace(bMin,bMax,300)
        some_ones = np.ones_like(B)
        for hLvl in hSet:
            X = self.solution[t].PolicyFunc.xFunc(B,hLvl*some_ones,MedShk*some_ones)
            plt.plot(B,X)
        plt.xlabel('Market resources bLvl')
        plt.ylabel('Expenditure level xLvl')
        plt.show()

        
    def plotxFuncByMedShk(self,t,hLvl,bMin=None,bMax=20.0,ShkSet=None):
        '''
        Plot the expenditure function vs bLvl at a fixed health level and
        a set of medical need shock values.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        None
        '''
        if ShkSet is None:
            ShkSet = self.solution[t].PolicyFunc.xFunc.z_list
        if bMin is None:
            bMin = self.solution[t].PolicyFunc.xFunc.x_list[0]
            
        B = np.linspace(bMin,bMax,300)
        some_ones = np.ones_like(B)
        for MedShk in ShkSet:
            X = self.solution[t].PolicyFunc.xFunc(B,hLvl*some_ones,MedShk*some_ones)
            plt.plot(B,X)
        plt.xlabel('Market resources bLvl')
        plt.ylabel('Expenditure level xLvl')
        plt.show()
        
        
    def plotcFuncByHealth(self,t,MedShk,bMin=None,bMax=20.0,hSet=None):
        '''
        Plot the consumption function vs bLvl at a fixed medical need shock and
        a set of health values.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        None
        '''
        if hSet is None:
            hSet = self.solution[t].PolicyFunc.xFunc.y_list
        if bMin is None:
            bMin = self.solution[t].PolicyFunc.xFunc.x_list[0]
            
        B = np.linspace(bMin,bMax,300)
        some_ones = np.ones_like(B)
        for hLvl in hSet:
            C = self.solution[t].PolicyFunc.cFunc(B,hLvl*some_ones,MedShk*some_ones)
            plt.plot(B,C)
        plt.xlabel('Market resources bLvl')
        plt.ylabel('Consumption level cLvl')
        plt.show()

        
    def plotcFuncByMedShk(self,t,hLvl,bMin=None,bMax=20.0,ShkSet=None):
        '''
        Plot the consumption function vs bLvl at a fixed health level and
        a set of medical need shock values.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        None
        '''
        if ShkSet is None:
            ShkSet = self.solution[t].PolicyFunc.xFunc.z_list
        if bMin is None:
            bMin = self.solution[t].PolicyFunc.xFunc.x_list[0]
            
        B = np.linspace(bMin,bMax,300)
        some_ones = np.ones_like(B)
        for MedShk in ShkSet:
            C = self.solution[t].PolicyFunc.cFunc(B,hLvl*some_ones,MedShk*some_ones)
            plt.plot(B,C)
        plt.xlabel('Market resources bLvl')
        plt.ylabel('Consumption level cLvl')
        plt.show()


    def plotiFuncByHealth(self,t,MedShk,bMin=None,bMax=20.0,hSet=None):
        '''
        Plot the investment function vs bLvl at a fixed medical need shock and
        a set of health values.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        None
        '''
        if hSet is None:
            hSet = self.solution[t].PolicyFunc.xFunc.y_list
        if bMin is None:
            bMin = self.solution[t].PolicyFunc.xFunc.x_list[0]
            
        B = np.linspace(bMin,bMax,300)
        some_ones = np.ones_like(B)
        for hLvl in hSet:
            I = self.solution[t].PolicyFunc.iFunc(B,hLvl*some_ones,MedShk*some_ones)
            plt.plot(B,I)
        plt.xlabel('Market resources bLvl')
        plt.ylabel('Investment level iLvl')
        plt.show()

        
    def plotiFuncByMedShk(self,t,hLvl,bMin=None,bMax=20.0,ShkSet=None):
        '''
        Plot the investment function vs bLvl at a fixed health level and
        a set of medical need shock values.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        None
        '''
        if ShkSet is None:
            ShkSet = self.solution[t].PolicyFunc.xFunc.z_list
        if bMin is None:
            bMin = self.solution[t].PolicyFunc.xFunc.x_list[0]
            
        B = np.linspace(bMin,bMax,300)
        some_ones = np.ones_like(B)
        for MedShk in ShkSet:
            I = self.solution[t].PolicyFunc.iFunc(B,hLvl*some_ones,MedShk*some_ones)
            plt.plot(B,I)
        plt.xlabel('Market resources bLvl')
        plt.ylabel('Investment level iLvl')
        plt.show()
        
        
    def plotvFuncByHealth(self,t,bMin=None,bMax=20.0,hSet=None,pseudo_inverse=False):
        '''
        Plot the value function vs bLvl at a set of health values.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        None
        '''
        if hSet is None:
            hSet = self.solution[t].vFunc.vNvrsFunc.y_list
        if bMin is None:
            bMin = self.solution[t].vFunc.vNvrsFunc.x_list[0]
            
        B = np.linspace(bMin,bMax,300)
        some_ones = np.ones_like(B)
        for hLvl in hSet:
            if pseudo_inverse:
                V = self.solution[t].vFunc.vNvrsFunc(B,hLvl*some_ones)
            else:
                V = self.solution[t].vFunc(B,hLvl*some_ones)
            plt.plot(B,V)
        plt.xlabel('Market resources bLvl')
        plt.ylabel('Value v')
        plt.show()

        
    def plotdvdbFuncByHealth(self,t,bMin=None,bMax=20.0,hSet=None,pseudo_inverse=False):
        '''
        Plot the marginal value function with respect to market resources vs bLvl
        at a set of health values.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        None
        '''
        if hSet is None:
            hSet = self.solution[t].dvdbFunc.cFunc.y_list
        if bMin is None:
            bMin = self.solution[t].dvdbFunc.cFunc.x_list[0]
            
        B = np.linspace(bMin,bMax,300)
        some_ones = np.ones_like(B)
        for hLvl in hSet:
            if pseudo_inverse:
                dvdb = self.solution[t].dvdbFunc.cFunc(B,hLvl*some_ones)
            else:
                dvdb = self.solution[t].dvdbFunc(B,hLvl*some_ones)
            plt.plot(B,dvdb)
        plt.xlabel('Market resources bLvl')
        plt.ylabel('Marginal value dvdb')
        plt.show()

        
    def plotdvdhFuncByHealth(self,t,bMin=None,bMax=20.0,hSet=None,Alt=False):
        '''
        Plot the marginal value function with respect to health status vs bLvl
        at a set of health values.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        None
        '''
        if hSet is None:
            hSet = self.solution[t].dvdhFunc.y_list
        if bMin is None:
            bMin = self.solution[t].dvdhFunc.x_list[0]
            
        B = np.linspace(bMin,bMax,300)
        some_ones = np.ones_like(B)
        for hLvl in hSet:
            if Alt:
                dvdh = self.solution[t].dvdhFuncAlt(B,hLvl*some_ones)
            else:
                dvdh = self.solution[t].dvdhFunc(B,hLvl*some_ones)
            plt.plot(B,dvdh)
        plt.xlabel('Market resources bLvl')
        plt.ylabel('Marginal value dvdh')
        plt.show()        
    
        
        
if __name__ == '__main__':
    from time import clock
    from HARKutilities import plotFuncs
    import HealthInvParams as Params
    
    t_start = clock()
    TestType = HealthInvestmentConsumerType(**Params.test_params)
    TestType.makeQuarticIncomePath()
    TestType.makeConstantMedPrice()
    TestType.update()
    t_end = clock()
    print('Making a health investment consumer took ' + str(t_end-t_start) + ' seconds.')
    
    t_start = clock()
    TestType.solve()
    t_end = clock()
    print('Solving a health investment consumer took ' + str(t_end-t_start) + ' seconds.')
    
    t=0
    bMax=200.
    
    TestType.plotxFuncByHealth(t,MedShk=1.0,bMax=bMax)
    TestType.plotxFuncByMedShk(t,hLvl=0.7,bMax=bMax)
    
    TestType.plotcFuncByHealth(t,MedShk=1.0,bMax=bMax)
    TestType.plotcFuncByMedShk(t,hLvl=0.7,bMax=bMax)
    
    TestType.plotiFuncByHealth(t,MedShk=1.0,bMax=bMax)
    TestType.plotiFuncByMedShk(t,hLvl=0.7,bMax=bMax)
    
    TestType.plotvFuncByHealth(t,pseudo_inverse=False,bMax=bMax)
    TestType.plotdvdbFuncByHealth(t,pseudo_inverse=False,bMax=bMax)
    TestType.plotdvdhFuncByHealth(t,bMax=bMax)
    TestType.plotdvdhFuncByHealth(t,bMax=bMax,Alt=True)

    TestType.T_sim = 25
    TestType.AgentCount = 10000
    TestType.track_vars = ['cLvlNow','MedLvlNow','iLvlNow','hLvlNow','aLvlNow','xLvlNow']
    TestType.aLvlInit = np.random.rand(10000)*5. + 3.
    TestType.HlvlInit = np.random.rand(10000)*0.45 + 0.5
    BornArray = np.zeros((25,10000),dtype=bool)
    BornArray[0,:] = True
    TestType.BornBoolArray = BornArray
    TestType.initializeSim()
    
    t_start = clock()
    TestType.simulate()
    t_end = clock()
    print('Simulating ' + str(TestType.AgentCount) + ' health investment consumers took ' + str(t_end-t_start) + ' seconds.')
    
#    for t in range(25):
#        TestType.plotxFuncByHealth(t,MedShk=1.0,bMax=bMax)

    
