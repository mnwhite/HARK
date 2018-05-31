'''
This module contains the solver and agent class for the model for "...An Ounce of Prevention..."
'''

import sys
import os
sys.path.insert(0,'../')
sys.path.insert(0,'../ConsumptionSaving/')

import numpy as np
from copy import copy
from scipy.stats import norm
from scipy.optimize import brentq
from scipy.special import erfc
from HARKcore import NullFunc, Solution, HARKobject, AgentType
from HARKinterpolation import ConstantFunction, LinearInterp, CubicInterp, BilinearInterp, TrilinearInterp, LinearInterpOnInterp1D, UpperEnvelope, LowerEnvelope
from HARKutilities import makeGridExpMult, CRRAutility, CRRAutilityP, CRRAutilityP_inv, CRRAutility_inv, plotFuncs
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
    
    def derivativeX(self,b,h):
        return CRRAutilityP(self.vNvrsFunc(b,h),gam=self.CRRA)*self.vNvrsFunc.derivativeX(b,h)
    
    
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
    def __init__(self, xFunc, iFunc, bFromxFunc, CopayFunc, MedShkMeanFunc, MedShkStdFunc, MedPrice):
        '''
        Constructor method for a new instance of HealthInvestmentPolicyFunc.
        
        Parameters
        ----------
        xFunc : function
            Expenditure function (cLvl & MedLvl), defined over (bLvl,hLvl,Dev).
        iFunc : function
            Health investment function, defined over (bLvl,hLvl,Dev).
        bFromxFunc : function
            Transformed consumption share function, defined over (xLvl,MedShkAdj).  Badly named.
        CopayFunc : function
            Coinsurance rate for medical care as a function of hLvl.
        MedShkMeanFunc : function
            Mean of log medical need shocks as a function of hLvl.
        MedShkStdFunc : function
            Stdev of log medical need shocks as a function of hLvl.
        MedPrice : float
            Relative price of a unit of medical care.
        
        Returns
        -------
        None
        '''
        self.xFunc = xFunc
        self.iFunc = iFunc
        self.bFromxFunc = bFromxFunc
        self.CopayFunc = CopayFunc
        self.MedShkMeanFunc = MedShkMeanFunc
        self.MedShkStdFunc = MedShkStdFunc
        self.MedPrice = MedPrice
        
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
        Dev = (MedShk - self.MedShkMeanFunc(hLvl))/self.MedShkStdFunc(hLvl)
        xLvl = self.xFunc(bLvl,hLvl,Dev)
        CopayEff = self.CopayFunc(hLvl)*self.MedPrice
        cShareTrans = self.bFromxFunc(xLvl,MedShk*CopayEff)
        q = np.exp(-cShareTrans)
        cLvl = xLvl/(1.+q)
        MedLvl = xLvl/CopayEff*q/(1.+q)
        iLvl = self.iFunc(bLvl,hLvl,Dev)
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
        
        
class TransConShareFunc(HARKobject):
    '''
    A class for representing the "transformed consumption share" function with
    "double CRRA" utility.  Instances of this class take as inputs xLvl and MedShkEff
    and return a transformed consumption share b.  By definition, optimal consumption
    cLvl = xLvl/(1 + exp(-b)).
    '''
    distance_criteria = ['CRRAcon','CRRAmed']
    
    def __init__(self,CRRAcon,CRRAmed):
        Bhalf = makeGridExpMult(0.0, 30.0, 40, 2)
        B = np.unique(np.concatenate([-Bhalf,Bhalf]))
        f = lambda b : b + (1. - CRRAcon/CRRAmed)*np.log(1. + np.exp(-b))
        fp = lambda b : 1. - (1. - CRRAcon/CRRAmed)*np.exp(-b)/(1. + np.exp(-b))
        G = f(B)
        D = 1./fp(B)
        f_inv = CubicInterp(G,B,D, lower_extrap=True)
        
        self.CRRAcon = CRRAcon
        self.CRRAmed = CRRAmed
        self.f_inv = f_inv
        self.coeff_x = (1. - CRRAcon/CRRAmed)
        self.coeff_Shk = (1. - 1./CRRAmed)
        
    def __call__(self,xLvl,MedShkEff):
        '''
        Evaluate the "transformed consumption share" function.
        
        Parameters
        ----------
        xLvl : np.array
            Array of expenditure levels.
        MedShkEff : np.array
             Identically shaped array of effective medical need shocks: MedShk*MedPrice*Copay.
             
        Returns
        -------
        b : np.array
            Identically shaped array of transformed consumption shares.
        '''
        b = self.f_inv(self.coeff_x*np.log(xLvl) - self.coeff_Shk*np.log(MedShkEff))
        return b



def solveHealthInvestment(solution_next,CRRA,DiscFac,CRRAmed,IncomeNext,IncomeNow,Rfree,Cfloor,LifeUtility,
                          Bequest0,Bequest1,MedPrice,aXtraGrid,bLvlGrid,Hcount,hLvlGrid,
                          HealthProdFunc,MargHealthProdInvFunc,HealthShkStd0,HealthShkStd1,
                          ExpHealthNextFunc,ExpHealthNextInvFunc,LivPrbFunc,bFromxFunc,PremiumFunc,CopayFunc,
                          MedShkMeanFunc,MedShkStdFunc,MedShkCount,ConvexityFixer,SubsidyFunc,CalcExpectationFuncs):
    '''
    Solves the one period problem in the health investment model.
    
    Parameters
    ----------
    solution_next : HealthInvSolution
        Solution to next period's problem in the same model.  Should have attributes called
        [list of attribute names here].
    CRRA : float
        Coefficient of relative risk aversion for consumption.
    DiscFac : float
        Intertemporal discount factor.
    CRRAmed : float
        Coefficient of relative risk aversion for medical care.
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
    Hcount : int
        Number of exogenous gridpoints in the post-health state grid.
    hLvlGrid : np.array
        Exogenous grid of beginning-of-period health levels for use in the JDfix step.
    HealthProdFunc : function
        Additional health produced as a function of iLvl.
    MargHealthProdInvFunc : function
        Inverse of marginal health produced function.  Takes a marginal health produced,
        returns an iLvl with that marginal productivity.
    HealthShkStd0 : float
        Base standard deviation of health shocks.
    HealthShkStd1 : float
        Change in stdev of health shocks with health.
    ExpHealthNextFunc : function
        Expected pre-investment health as a function of current health.
    ExpHealthNextInvFunc : function
        Current health as a function of expected pre-investment health.
    LivPrbFunc : function
        Survival probability to next period as a function of end-of-period health.
    bFromxFunc : function
        Transformed consumption share as a function of expenditure and effective medical need.
    PremiumFunc : function
        Out-of-pocket premiums as a function of health.
    CopayFunc : function
        Out-of-pocket coinsurance rate as a function of health.
    MedShkMeanFunc : function
        Mean of log medical need shock as a function of health.
    MedShkStdFunc : function
        Stdev of log medical need shock as a function of health.
    MedShkCount : int
        Number of non-zero medical need shocks to use in EGM step.
    ConvexityFixer : JDfixer
        Instance of JDfixer that transforms irregular data grids onto exog grids.
    SubsidyFunc : function
        Function that gives health investment subsidy as a function of health.
    CalcExpectationFuncs : bool
        Indicator for whether to calculate PDV of expected lifetime medical care
        and life expectancy.
        
    Returns
    -------
    solution_now : HealthInvSolution
        Solution to this period's health investment problem.
    '''
    # Define utility functions
    u = lambda C : CRRAutility(C,gam=CRRA)
    uP = lambda C : CRRAutilityP(C,gam=CRRA)
    uPinv = lambda Up : CRRAutilityP_inv(Up,gam=CRRA)
    BequestMotive = lambda a : Bequest1*(CRRAutility(a + Bequest0,gam=CRRA) - CRRAutility(Bequest0,gam=CRRA))
    BequestMotiveP = lambda a : Bequest1*CRRAutilityP(a + Bequest0,gam=CRRA)
    u0 = -u(LifeUtility)
    uMed = lambda M : CRRAutility(M,gam=CRRAmed)
    
    # Make a grid of post-state health, making sure there are no levels of health
    # that would never be reached by the EGM procedure
    Hmin = np.minimum(0.0,ExpHealthNextFunc(0.0)-0.001) # Span the bottom
    Hmax = np.maximum(1.0,ExpHealthNextFunc(1.0)+HealthProdFunc(100.)+0.001) # Span the top
    Hgrid = np.linspace(Hmin,Hmax,num=Hcount)

    # Unpack next period's solution
    vFuncNext = solution_next.vFunc
    dvdbFuncNext = solution_next.dvdbFunc
    
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
    vNext_tiled = np.tile(vNext,(1,Hcount,1))
    dvdbNext_tiled = np.tile(dvdbNext,(1,Hcount,1))
    BequestMotiveArray = np.tile(np.reshape(BequestMotive(aLvlGrid),(aCount,1)),(1,Hcount))
    BequestMotiveParray = np.tile(np.reshape(BequestMotiveP(aLvlGrid),(aCount,1)),(1,Hcount))
        
    # Calculate the probability of arriving at each future health state from each current post-health state
    Harray_temp = np.tile(np.reshape(Hgrid,(1,Hcount,1)),(1,1,hNextCount))
    hNextArray_temp = np.tile(np.reshape(hNextGrid,(1,1,hNextCount)),(1,Hcount,1))
    HealthShkStd = HealthShkStd0 + HealthShkStd1*Harray_temp
    zArray = (hNextArray_temp - Harray_temp)/HealthShkStd
    ProbArray = np.zeros((1,Hcount,hNextCount))
    BaseProbs = norm.pdf(zArray[:,:,1:-1]) # Don't include h=0 or h=1
    ProbSum = np.tile(np.sum(BaseProbs,axis=2,keepdims=True),(1,1,hNextCount-2))
    LivPrb = norm.sf(LivPrbFunc(Hgrid))
    TerriblePrb = norm.cdf(zArray[:,:,0])
    PerfectPrb = norm.sf(zArray[:,:,-1])
    MainProbs = BaseProbs/ProbSum*np.tile(np.reshape((1.0-TerriblePrb-PerfectPrb),(1,Hcount,1)),(1,1,hNextCount-2))
    ProbArray[:,:,1:-1] = MainProbs
    ProbArray[:,:,0] = TerriblePrb
    ProbArray[:,:,-1] = PerfectPrb
    ProbArray *= np.tile(np.reshape(LivPrb,(1,Hcount,1)),(1,1,hNextCount)) # Adjust probabilities for survival prob
    DiePrb = 1. - LivPrb
        
    # Calculate the rate of change in probabilities of arriving in each future health state from end-of-period health
    H_eps = 0.0001 # Perturb end-of-period health by just a bit
    Harray_temp = np.tile(np.reshape(Hgrid+H_eps,(1,Hcount,1)),(1,1,hNextCount))
    HealthShkStd = HealthShkStd0 + HealthShkStd1*Harray_temp
    zArray = (hNextArray_temp - Harray_temp)/HealthShkStd
    ProbArrayAlt = np.zeros((1,Hcount,hNextCount))
    BaseProbs = norm.pdf(zArray[:,:,1:-1]) # Don't include h=0 or h=1
    ProbSum = np.tile(np.sum(BaseProbs,axis=2,keepdims=True),(1,1,hNextCount-2))
    LivPrbAlt = norm.sf(LivPrbFunc(Hgrid+H_eps))
    TerriblePrb = norm.cdf(zArray[:,:,0])
    PerfectPrb = norm.sf(zArray[:,:,-1])
    MainProbsAlt = BaseProbs/ProbSum*np.tile(np.reshape((1.0-TerriblePrb-PerfectPrb),(1,Hcount,1)),(1,1,hNextCount-2))
    ProbArrayAlt[:,:,1:-1] = MainProbsAlt
    ProbArrayAlt[:,:,0] = TerriblePrb
    ProbArrayAlt[:,:,-1] = PerfectPrb
    ProbArrayAlt *= np.tile(np.reshape(LivPrbAlt,(1,Hcount,1)),(1,1,hNextCount)) # Adjust probabilities for survival prob
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
    MargValueRatioAdj[np.isinf(MargValueRatioAdj)] = 0.0 # Code infinities as zero
    
    if CalcExpectationFuncs:
        # Evaluate PDV of future expected lifetime medical care and life expectancy, and make interpolations
        if hasattr(solution_next,'TotalMedPDVfunc'):
            FutureMed = Rfree**(-1.)*np.sum(solution_next.TotalMedPDVfunc(bNextArray,hNextArray)*ProbArray_tiled,axis=2)
            FutureOOP = Rfree**(-1.)*np.sum(solution_next.OOPmedPDVfunc(bNextArray,hNextArray)*ProbArray_tiled,axis=2)
            FutureLife = np.sum(solution_next.ExpectedLifeFunc(bNextArray,hNextArray)*ProbArray_tiled,axis=2)
            FutureSubsidy = Rfree**(-1.)*np.sum(solution_next.SubsidyPDVfunc(bNextArray,hNextArray)*ProbArray_tiled,axis=2)
            FutureMedicare = Rfree**(-1.)*np.sum(solution_next.MedicarePDVfunc(bNextArray,hNextArray)*ProbArray_tiled,axis=2)
            FutureWelfare = Rfree**(-1.)*np.sum(solution_next.WelfarePDVfunc(bNextArray,hNextArray)*ProbArray_tiled,axis=2)
        else: # This gets triggered in the first (last) period
            FutureMed = np.zeros_like(DiePrb_tiled)
            FutureOOP = np.zeros_like(DiePrb_tiled)
            FutureLife = np.zeros_like(DiePrb_tiled)
            FutureSubsidy = np.zeros_like(DiePrb_tiled)
            FutureMedicare = np.zeros_like(DiePrb_tiled)
            FutureWelfare = np.zeros_like(DiePrb_tiled)
        FutureTotalMedFunc = BilinearInterp(FutureMed,aLvlGrid,Hgrid)
        FutureOOPmedFunc = BilinearInterp(FutureOOP,aLvlGrid,Hgrid)
        FutureLifeFunc = BilinearInterp(FutureLife,aLvlGrid,Hgrid)
        FutureSubsidyFunc = BilinearInterp(FutureSubsidy,aLvlGrid,Hgrid)
        FutureMedicareFunc = BilinearInterp(FutureMedicare,aLvlGrid,Hgrid)
        FutureWelfareFunc = BilinearInterp(FutureWelfare,aLvlGrid,Hgrid)        
    
    # Store end-of-period marginal value functions
    EndOfPrdvFunc = BilinearInterp(EndOfPrdv,aLvlGrid,Hgrid)
    dvdaNvrs = uPinv(EndOfPrddvda)
    dvdaNvrsFunc = BilinearInterp(dvdaNvrs,aLvlGrid,Hgrid)
    dvdaFunc = MargValueFunc2D(dvdaNvrsFunc,CRRA)
    dvdHfunc = BilinearInterp(EndOfPrddvdH,aLvlGrid,Hgrid)
    
    # Use a fixed point loop to find optimal health investment (unconstrained).
    # Coinsurance rate appears in the FOC for iLvl, but it depends on hLvl.
    # Need to do a search for iLvl,hLvl that satisfies the FOC and transition equation.
    tol = 1e-5
    LoopCount = 0
    MaxLoops = 20
    diff = np.ones_like(MargValueRatio)
    these = diff > tol
    points_left = np.sum(these)
    iNow = np.zeros_like(Harray)
    hNow = ExpHealthNextInvFunc(Harray) # Initial guess of iLvl=0 implies this hLvl
    while (points_left > 0) and (LoopCount < MaxLoops):
        Ratio = MargValueRatioAdj[these]
        H = Harray[these]
        hGuess = hNow[these].flatten()
        CopayGuess = CopayFunc(hGuess)
        iGuess = np.maximum(MargHealthProdInvFunc(CopayGuess*Ratio),0.0) # Invert FOC for iLvl
        iGuess[Ratio == 0.] = 0.0 # Fix "bad" iLvl values
        temp = np.logical_and(Ratio > 0., np.logical_not(np.isinf(Ratio)))
        iGuess[temp] = np.maximum(iGuess[temp], SubsidyFunc(hGuess[temp])/MedPrice) # If i is good but not using all subsidy, use it all
        hNow[these] = ExpHealthNextInvFunc(H - HealthProdFunc(iGuess)) # Update guess of hLvl
        diff[these] = np.abs(iGuess - iNow[these]) # Calculate difference and update guesses
        iNow[these] = iGuess
        these = diff > tol
        points_left = np.sum(these)
        LoopCount += 1
        
    # Make a grid of medical need values
    DevGrid = np.linspace(-3.01,5.01,MedShkCount) # Made this a little wider than DevGridDense
    DevArray = np.tile(np.reshape(DevGrid,(1,1,MedShkCount)),(aCount,Hcount,1)) # Standard deviations from mean of log MedShk
    LogShkMeanArray = np.tile(np.reshape(MedShkMeanFunc(hNow),(aCount,Hcount,1)),(1,1,MedShkCount)) # Mean of log medical shock
    LogShkStdArray = np.tile(np.reshape(MedShkStdFunc(hNow),(aCount,Hcount,1)),(1,1,MedShkCount)) # Stdev of log medical shock
    LogShkArray = LogShkMeanArray + DevArray*LogShkStdArray # Log of medical needs shock
    MedShkArrayBig = np.exp(LogShkArray)
    DevGridDense = np.linspace(-3.,5.,MedShkCount*3+1) # remove +1 later?
    
    # Make 3D arrays of states, health investment, insurance terms, and (marginal) values
    aLvlArrayBig = np.tile(np.reshape(aLvlGrid,(aCount,1,1)),(1,Hcount,MedShkCount))
    hLvlArrayBig = np.tile(np.reshape(hNow,(aCount,Hcount,1)),(1,1,MedShkCount))
    iLvlArrayBig = np.tile(np.reshape(iNow,(aCount,Hcount,1)),(1,1,MedShkCount))
    PremiumArrayBig = PremiumFunc(hLvlArrayBig)
    CopayArrayBig = CopayFunc(hLvlArrayBig)
    EndOfPrdvBig = np.tile(np.reshape(EndOfPrdv,(aCount,Hcount,1)),(1,1,MedShkCount))
    EndOfPrddvdaBig = np.tile(np.reshape(EndOfPrddvda,(aCount,Hcount,1)),(1,1,MedShkCount))
    EndOfPrddvdHBig = np.tile(np.reshape(EndOfPrddvdH,(aCount,Hcount,1)),(1,1,MedShkCount))
      
    # Use the first order conditions to calculate optimal expenditure on consumption and mitigative care (unconstrained)
    EndOfPrddvdaNvrs = uPinv(EndOfPrddvdaBig)
    cLvlArrayBig = EndOfPrddvdaNvrs
    EffPriceArrayBig = MedPrice*CopayArrayBig
    MedLvlArrayBig = EffPriceArrayBig**(-1./CRRAmed)*MedShkArrayBig**(1.-1./CRRAmed)*cLvlArrayBig**(CRRA/CRRAmed)
    xLvlArrayBig = cLvlArrayBig + EffPriceArrayBig*MedLvlArrayBig
    iCostArrayBig = CopayArrayBig*np.maximum(MedPrice*iLvlArrayBig - SubsidyFunc(hLvlArrayBig), 0.0)
    bLvlArrayBig = aLvlArrayBig + xLvlArrayBig + iCostArrayBig + PremiumArrayBig
    vArrayBig = u(cLvlArrayBig) + uMed(MedLvlArrayBig/MedShkArrayBig) + u0 + EndOfPrdvBig
    dvdhArrayBig = ExpHealthNextFunc.der(hLvlArrayBig)*EndOfPrddvdHBig
          
    # Make an exogenous grid of xLvl and MedShk values where individual is constrained.
    # The idea here is that when the individual is constrained by aLvl=0, both xLvl and
    # iLvl will be non-decreasing in bLvl.  So we fix xLvl exogenously and then find the
    # hLvl, iLvl that justify spending that much.
    bCnstCount = 16
    DevArrayCnst = np.tile(np.reshape(DevGrid,(1,1,MedShkCount)),(bCnstCount,Hcount,1))
    FractionGrid = np.tile(np.reshape(np.linspace(0.,1.,bCnstCount,endpoint=False),(bCnstCount,1,1)),(1,Hcount,MedShkCount))
    xLvlCnstMin = np.minimum(0.01*xLvlArrayBig[0,:,:], 0.01)
    xLvlArrayCnst = np.tile(np.reshape(xLvlArrayBig[0,:,:] - xLvlCnstMin,(1,Hcount,MedShkCount)),(bCnstCount,1,1))*FractionGrid + xLvlCnstMin
    HarrayCnst = np.tile(np.reshape(Hgrid,(1,Hcount,1)),(bCnstCount,1,MedShkCount))
    EndOfPrddvdHCnst = np.tile(np.reshape(EndOfPrddvdH[0,:],(1,Hcount,1)),(bCnstCount,1,MedShkCount))
    EndOfPrddvdHCnstAdj = np.maximum(EndOfPrddvdHCnst,0.0)
    EndOfPrddvdHCnstAdj[np.isinf(EndOfPrddvdHCnstAdj)] = 0.0 # code infinities as zero
    
    # Use a fixed point loop to solve for the constrained solution at each constrained xLvl
    tol = 1e-5
    LoopCount = 0
    MaxLoops = 20
    diff = np.ones_like(HarrayCnst)
    these = diff > tol
    points_left = np.sum(these)
    hCnst = np.tile(np.reshape(hLvlArrayBig[0,:,:],(1,Hcount,MedShkCount)),(bCnstCount,1,1))
    iCnst = np.zeros_like(hCnst)
    while (points_left > 0) and (LoopCount < MaxLoops):
        hGuess = hCnst[these] # Get current guess of beginning of period health
        x = xLvlArrayCnst[these] # Get relevant expenditure values
        EffPrice = MedPrice*CopayFunc(hGuess) # Calculate effective coinsurance rate
        Shk = np.exp(MedShkMeanFunc(hGuess) + DevArrayCnst[these]*MedShkStdFunc(hGuess))
        q = np.exp(-bFromxFunc(x,EffPrice*Shk)) # Get transformed consumption shares
        cLvl = x/(1. + q) # Calculate consumption given expenditure and transformed consumption share
        dvdH = EndOfPrddvdHCnstAdj[these] # Get relevant end of period marginal value of post-health
        ImpliedMargHealthProd = uP(cLvl)*EffPrice/dvdH # Calculate what the marginal product of health investment must be for cLvl to be optimal
        iLvl = np.maximum(MargHealthProdInvFunc(ImpliedMargHealthProd),0.0) # Invert FOC for iLvl when constrained
        iLvl[dvdH == 0.] = 0. # Fix zero dvdH (actually negative) to have zero investment
        iLvl[x == 0.] = 0. # Solution when x=0 is also i=0
        temp = dvdH > 0.
        iLvl[temp] = np.maximum(iLvl[temp], SubsidyFunc(hGuess[temp])/MedPrice) # If i is good but not using all subsidy, use it all
        hCnst[these] = ExpHealthNextInvFunc(HarrayCnst[these] - HealthProdFunc(iLvl)) # Update constrained hLvl
        diff[these] = np.abs(iCnst[these] - iLvl) # Calculate distance between old and new iLvl
        iCnst[these] = iLvl
        these = diff > tol
        points_left = np.sum(these) # Move to next iteration
        LoopCount += 1
        
        # NEED TO ADJUST BLOCK ABOVE TO HANDLE SUBSIDY WHEN CONSTRAINED!!
        
    # Rename the constrained arrays and calculate (marginal) values for them
    hLvlArrayCnst = hCnst
    iLvlArrayCnst = iCnst
    EffPriceCnst = MedPrice*CopayFunc(hCnst)
    MedShkArrayCnst = np.exp(MedShkMeanFunc(hCnst) + DevArrayCnst*MedShkStdFunc(hCnst))
    bLvlArrayCnst = 0.0 + xLvlArrayCnst + EffPriceCnst*iCnst + PremiumFunc(hCnst)
    q = np.exp(-bFromxFunc(xLvlArrayCnst,EffPriceCnst*MedShkArrayCnst)) 
    cCnst = xLvlArrayCnst/(1. + q)
    MedCnst = q*cCnst/EffPriceCnst
    vArrayCnst = u(cCnst) + uMed(MedCnst/MedShkArrayCnst) + u0 + np.tile(np.reshape(EndOfPrdvBig[0,:,:],(1,Hcount,MedShkCount)),(bCnstCount,1,1))
    dvdhArrayCnst = ExpHealthNextFunc.der(hLvlArrayCnst)*EndOfPrddvdHCnst
    
    # Combine the constrained and unconstrained solutions into unified arrays
    bLvlArrayAll = np.concatenate((bLvlArrayCnst,bLvlArrayBig),axis=0)
    hLvlArrayAll = np.concatenate((hLvlArrayCnst,hLvlArrayBig),axis=0)
    DevArrayAll = np.concatenate((DevArray,DevArrayCnst),axis=0)
    xLvlArrayAll = np.concatenate((xLvlArrayCnst,xLvlArrayBig),axis=0)
    iLvlArrayAll = np.concatenate((iLvlArrayCnst,iLvlArrayBig),axis=0)
    vArrayAll = np.concatenate((vArrayCnst,vArrayBig),axis=0)
    dvdhArrayAll = np.concatenate((dvdhArrayCnst,dvdhArrayBig),axis=0)
    
    # Apply the Jorgensen-Druedahl convexity fix and construct expenditure and investment functions
    xLvlArray, iLvlArray, vNvrsArray, dvdhArray = ConvexityFixer(bLvlArrayAll,hLvlArrayAll,DevArrayAll,
                                        vArrayAll,dvdhArrayAll,xLvlArrayAll,iLvlArrayAll,bLvlGrid,hLvlGrid,DevGridDense)
    xFuncNow = TrilinearInterp(xLvlArray,bLvlGrid,hLvlGrid,DevGridDense)
    iFuncNow = TrilinearInterp(iLvlArray,bLvlGrid,hLvlGrid,DevGridDense)
    PolicyFuncNow = HealthInvestmentPolicyFunc(xFuncNow,iFuncNow,bFromxFunc,CopayFunc,MedShkMeanFunc,MedShkStdFunc,MedPrice)
    bLvlCount = bLvlGrid.size
    hLvlCount = hLvlGrid.size
    ShkCount  = DevGridDense.size
    
    # Make an array of values that are attained if we hit the Cfloor this period
    H_temp       = ExpHealthNextFunc(hLvlGrid)
    EndOfPrdvFunc_no_assets = LinearInterp(Hgrid,EndOfPrdv[0,:])
    EndOfPrdv_temp = np.tile(np.reshape(EndOfPrdvFunc_no_assets(H_temp),(1,hLvlCount,1)),(1,1,ShkCount))
    
    # Find CritDev using a clever closed formula
    b_temp = np.tile(np.reshape(bLvlGrid,(bLvlCount,1)),(1,hLvlCount))
    p_temp = np.tile(np.reshape(MedPrice*CopayFunc(hLvlGrid),(1,hLvlCount)),(bLvlCount,1))
    m_temp = np.tile(np.reshape(MedShkMeanFunc(hLvlGrid),(1,hLvlCount)),(bLvlCount,1))
    s_temp = np.tile(np.reshape(MedShkStdFunc(hLvlGrid),(1,hLvlCount)),(bLvlCount,1))
    LogCritShkAlt = (CRRAmed - CRRA)/(CRRAmed - 1.)*np.log(Cfloor) + CRRAmed/(CRRAmed - 1.)*np.log(b_temp/Cfloor - 1.) - np.log(p_temp)
    CritDevArray = (LogCritShkAlt - m_temp)/s_temp
    CritDevArray = np.minimum(np.maximum(CritDevArray,-3.0),5.0) # Apply bounds for grid
    CritDevFunc = BilinearInterp(CritDevArray,bLvlGrid,hLvlGrid)
    Never_Cfloor = CritDevArray == 5. # Critical shock happens higher than largest shock in grid (+5std)
    Always_Cfloor = CritDevArray == -3. # Critical shock happens lower than lowest shock in grid (-3std)
    CritShkPrbArray = norm.sf(CritDevArray)
    CritShkPrbArray[Never_Cfloor] = 0.
    CritShkPrbArray[Always_Cfloor] = 1.
    
    # Choose medical need shock grids for integration
    FracArray = np.tile(np.reshape(np.linspace(0.0,1.0,MedShkCount),(1,1,MedShkCount)),(bLvlCount,hLvlCount,1))
    zArray = -3.0 + FracArray*(np.tile(np.reshape(CritDevArray+3.0,(bLvlCount,hLvlCount,1)),(1,1,MedShkCount)))
    LogShkMeanArray = np.tile(np.reshape(MedShkMeanFunc(hLvlGrid),(1,hLvlCount,1)),(1,1,MedShkCount))
    LogShkStdArray  = np.tile(np.reshape(MedShkStdFunc(hLvlGrid),(1,hLvlCount,1)),(1,1,MedShkCount))
    MedShkValArray  = np.exp(LogShkMeanArray + zArray*LogShkStdArray)
    
    # Calculate probabilities of all of the medical shocks
    BasePrbArray = norm.pdf(zArray)
    SumPrbArray = np.sum(BasePrbArray,axis=2)
    AdjArray = np.tile(np.reshape((1.0-CritShkPrbArray)/SumPrbArray,(bLvlCount,hLvlCount,1)),(1,1,MedShkCount))
    MedShkPrbArray = BasePrbArray*AdjArray
    
    # Calculate expected value when hitting Cfloor (use truncated lognormal formula)
    mu = (1.-1./CRRAmed)*LogShkMeanArray[:,:,0] # underlying mean of lognormal shocks
    sigma = (1.-1./CRRAmed)*LogShkStdArray[:,:,0] # underlying std of lognormal shocks
    C1_vec = (MedPrice*CopayFunc(hLvlGrid))**(1.-1./CRRAmed)*Cfloor**(CRRA/CRRAmed - CRRA)/(1.-CRRAmed) # constant factor
    ExpectedAdjShkAtCfloor = -0.5*np.exp(mu+(sigma**2)*0.5)*(erfc(np.sqrt(0.5)*(sigma-CritDevArray))-2.0)/CritShkPrbArray
    X = ExpectedAdjShkAtCfloor*np.tile(np.reshape(C1_vec,(1,hLvlCount)),(bLvlCount,1)) # Don't know what to call this
    vFloor_expected = u(Cfloor) + u0 + EndOfPrdv_temp[:,:,0] + X
    vFloor_expected[Never_Cfloor] = 0.0
    
    # Find where each shock for integration falls on the DevGridDense
    IdxHi = np.minimum(np.searchsorted(DevGridDense,zArray),DevGridDense.size-1)
    IdxLo = IdxHi - 1
    DevLo = DevGridDense[IdxLo]
    DevHi = DevGridDense[IdxHi]
    alpha = (zArray - DevLo)/(DevHi - DevLo)
    alpha_comp = 1.0 - alpha
    bIdx  = np.tile(np.reshape(np.arange(bLvlCount),(bLvlCount,1,1)),(1,hLvlCount,MedShkCount))
    hIdx  = np.tile(np.reshape(np.arange(hLvlCount),(1,hLvlCount,1)),(bLvlCount,1,MedShkCount))
    
    # Calculate agents' controls and post-states on the grid of shocks
    x_temp = alpha_comp*xLvlArray[bIdx,hIdx,IdxLo] + alpha*xLvlArray[bIdx,hIdx,IdxHi]
    i_temp = alpha_comp*iLvlArray[bIdx,hIdx,IdxLo] + alpha*iLvlArray[bIdx,hIdx,IdxHi]
    i_temp = np.maximum(i_temp,0.0) # Prevent small glitchy negative values
    Copay_temp = np.tile(np.reshape(CopayFunc(hLvlGrid),(1,hLvlCount,1)),(bLvlCount,1,MedShkCount))
    Premium_temp = np.tile(np.reshape(PremiumFunc(hLvlGrid),(1,hLvlCount,1)),(bLvlCount,1,MedShkCount))
    EffPrice_temp = Copay_temp*MedPrice
    q_temp = np.exp(-bFromxFunc(x_temp,MedShkValArray*EffPrice_temp))
    c_temp = x_temp/(1. + q_temp)
    SubsidyMax = np.tile(np.reshape(SubsidyFunc(hLvlGrid),(1,hLvlCount,1)),(bLvlCount,1,MedShkCount))
    Subsidy_temp = np.minimum(i_temp*MedPrice,SubsidyMax)
    MedOOP_temp = q_temp*x_temp/(1. + q_temp)
    Med_temp = MedOOP_temp/EffPrice_temp
    iCost_temp = i_temp*EffPrice_temp - Subsidy_temp
    OOP_temp = MedOOP_temp + iCost_temp
    b_temp = np.tile(np.reshape(bLvlGrid,(bLvlCount,1,1)),(1,hLvlCount,MedShkCount))
    h_temp = np.tile(np.reshape(hLvlGrid,(1,hLvlCount,1)),(bLvlCount,1,MedShkCount))
    a_temp = np.maximum(b_temp - Premium_temp - x_temp - iCost_temp, 0.0)
    H_temp = ExpHealthNextFunc(h_temp) + HealthProdFunc(i_temp)
    v_temp = u(c_temp) + uMed(Med_temp/MedShkValArray) + u0 + EndOfPrdvFunc(a_temp,H_temp)
    
    # Integrate value according to the shock probabilities
    ValueArrayFlat = np.sum(v_temp*MedShkPrbArray,axis=2) + CritShkPrbArray*vFloor_expected
    
    # Integrate marginal value of bank balances according to the shock probabilities
    dvdb_temp = uP(c_temp)
    dvdbArray = np.sum(MedShkPrbArray*dvdb_temp, axis=2)
    dvdbNvrsArray = uPinv(dvdbArray)
    
    # Integrate marginal value of health according to the shock probabilities (NOT USED)
    dvdhArray = np.zeros_like(dvdbArray)
    
    # Make (marginal) value functions
    vNvrsFuncNow = BilinearInterp(ValueArrayFlat,bLvlGrid,hLvlGrid)
    vFuncNow = ValueFunc2D(vNvrsFuncNow,0.0,0.0)
    dvdbNvrsFuncNow = BilinearInterp(dvdbNvrsArray,bLvlGrid,hLvlGrid)
    dvdbFuncNow = MargValueFunc2D(dvdbNvrsFuncNow,CRRA)
    dvdhFuncNow = BilinearInterp(dvdhArray,bLvlGrid,hLvlGrid)
        
    # Package and return the solution object
    solution_now = HealthInvestmentSolution(PolicyFuncNow,vFuncNow,dvdbFuncNow,dvdhFuncNow)
    solution_now.dvdaFunc = dvdaFunc
    solution_now.dvdHfunc = dvdHfunc
    solution_now.CritDevFunc = CritDevFunc
    
    if CalcExpectationFuncs:
        # Calculate current values of various objects on the grid of shocks
        TotalMed_temp = MedPrice*(i_temp + Med_temp)
        Medicare_temp = (TotalMed_temp - Subsidy_temp)*(1.-Copay_temp)
        Welfare_temp = np.zeros_like(OOP_temp) # No welfare unless we hit Cfloor
        Life_temp = 2.0*norm.sf(LivPrbFunc(H_temp)) # Expected years we life this period

        # Calculate end-of-period states when at Cfloor
        a_Cfloor = np.zeros_like(b_temp[:,:,0])
        H_Cfloor = ExpHealthNextFunc(h_temp[:,:,0])
        
        # Calculate current values of various objects when at Cfloor
        Subsidy_Cfloor = np.zeros_like(a_Cfloor)
        OOP_Cfloor = np.maximum(b_temp[:,:,0] - Premium_temp[:,:,0] - Cfloor, 0.0)
        TotalMed_Cfloor = EffPrice_temp[:,:,0]**(-1./CRRAmed)*ExpectedAdjShkAtCfloor*Cfloor**(CRRA/CRRAmed)
        Cost_Cfloor = Cfloor + Premium_temp[:,:,0] + MedPrice*TotalMed_Cfloor # Total expected cost of all goods at Cfloor
        Medicare_Cfloor = (1.-Copay_temp[:,:,0])*OOP_Cfloor/Copay_temp[:,:,0] # Medicare pays for medical expenses until Cfloor is hit
        Welfare_Cfloor = Cost_Cfloor - b_temp[:,:,0] - Medicare_Cfloor # Welfare is whatever is not accounted for by bLvl or Medicare
        TotalMed_Cfloor[Never_Cfloor] = 0.0
        Welfare_Cfloor[Never_Cfloor] = 0.0
        Life_Cfloor = 2.0*norm.sf(LivPrbFunc(H_Cfloor))
        
        # Evaluate future PDV of various objects on the grid of shocks
        FutureTotalMed_temp = FutureTotalMedFunc(a_temp,H_temp)
        FutureOOPmed_temp = FutureOOPmedFunc(a_temp,H_temp)
        FutureLife_temp = FutureLifeFunc(a_temp,H_temp)
        FutureMedicare_temp = FutureMedicareFunc(a_temp,H_temp)
        FutureSubsidy_temp = FutureSubsidyFunc(a_temp,H_temp)
        FutureWelfare_temp = FutureWelfareFunc(a_temp,H_temp)
        
        # Calculate future PDV of various objects when we hit the Cfloor
        FutureTotalMed_Cfloor = FutureTotalMedFunc(a_Cfloor,H_Cfloor)
        FutureOOPmed_Cfloor = FutureOOPmedFunc(a_Cfloor,H_Cfloor)
        FutureLife_Cfloor = FutureLifeFunc(a_Cfloor,H_Cfloor)
        FutureMedicare_Cfloor = FutureMedicareFunc(a_Cfloor,H_Cfloor)
        FutureSubsidy_Cfloor = FutureSubsidyFunc(a_Cfloor,H_Cfloor)
        FutureWelfare_Cfloor = FutureWelfareFunc(a_Cfloor,H_Cfloor)
        
        # Calculate PDV of various objects from perspective of beginning of period
        TotalMedPDV = np.sum((TotalMed_temp + FutureTotalMed_temp)*MedShkPrbArray, axis=2) + (TotalMed_Cfloor + FutureTotalMed_Cfloor)*CritShkPrbArray
        OOPmedPDV = np.sum((OOP_temp + FutureOOPmed_temp)*MedShkPrbArray, axis=2) + (OOP_Cfloor + FutureOOPmed_Cfloor)*CritShkPrbArray
        MedicarePDV = np.sum((Medicare_temp + FutureMedicare_temp)*MedShkPrbArray, axis=2) + (Medicare_Cfloor + FutureMedicare_Cfloor)*CritShkPrbArray
        SubsidyPDV = np.sum((Subsidy_temp + FutureSubsidy_temp)*MedShkPrbArray, axis=2) + (Subsidy_Cfloor + FutureSubsidy_Cfloor)*CritShkPrbArray
        WelfarePDV = np.sum((Welfare_temp + FutureWelfare_temp)*MedShkPrbArray, axis=2) + (Welfare_Cfloor + FutureWelfare_Cfloor)*CritShkPrbArray
        ExpectedLife = np.sum((Life_temp + FutureLife_temp)*MedShkPrbArray, axis=2) + (Life_Cfloor + FutureLife_Cfloor)*CritShkPrbArray
        GovtPDV = MedicarePDV + SubsidyPDV + WelfarePDV
        
        # Make PDV functions and store them as attributes of the solution
        solution_now.TotalMedPDVfunc = BilinearInterp(TotalMedPDV,bLvlGrid,hLvlGrid)
        solution_now.OOPmedPDVfunc = BilinearInterp(OOPmedPDV,bLvlGrid,hLvlGrid)
        solution_now.MedicarePDVfunc = BilinearInterp(MedicarePDV,bLvlGrid,hLvlGrid)
        solution_now.SubsidyPDVfunc = BilinearInterp(SubsidyPDV,bLvlGrid,hLvlGrid)
        solution_now.WelfarePDVfunc = BilinearInterp(WelfarePDV,bLvlGrid,hLvlGrid)
        solution_now.ExpectedLifeFunc = BilinearInterp(ExpectedLife,bLvlGrid,hLvlGrid)
        solution_now.GovtPDVfunc = BilinearInterp(GovtPDV,bLvlGrid,hLvlGrid)
            
    return solution_now
    
    

class HealthInvestmentConsumerType(IndShockConsumerType):
    '''
    A class for representing agents in the health investment model.
    '''
    
    def __init__(self,**kwds):
        AgentType.__init__(self,solution_terminal=None,time_flow=True,pseudo_terminal=True,**kwds)
        self.time_inv = ['CRRA','DiscFac','MedCurve','Cfloor','LifeUtility','MargUtilityShift',
                         'Rfree','Bequest0','Bequest1','MedShkCount','HealthProd0','HealthProd1',
                         'HealthProd2','HealthShkStd0','HealthShkStd1','CalcExpectationFuncs']
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
            
        # Define boundaries of premiums and coinsurance rates
        PremiumMin = ConstantFunction(0.01)
        PremiumMax = ConstantFunction(10.) # This would never happen
        CopayMin = ConstantFunction(0.10)
        CopayMax = ConstantFunction(1.0) # Shouldn't happen
            
        PremiumFunc = []
        CopayFunc = []
        for t in range(self.T_cycle):
            Age = t*1
            y = self.IncomeNow[t]
            p0 = self.Premium0 + self.Sex*self.PremiumSex + self.PremiumAge*Age + self.PremiumAgeSq*Age**2 + self.PremiumInc*y + self.PremiumIncSq*y**2 + self.PremiumIncCu*y**3
            p1 = self.PremiumHealth + self.PremiumHealthAge*Age + self.PremiumHealthAgeSq*Age**2 + self.PremiumHealthInc*y + self.PremiumHealthIncSq*y**2
            p2 = self.PremiumHealthSq + self.PremiumHealthSqAge*Age + self.PremiumHealthSqAgeSq*Age**2 + self.PremiumHealthSqInc*y + self.PremiumHealthSqIncSq*y**2
            PremiumFunc.append(LowerEnvelope(UpperEnvelope(QuadraticFunction(p0,p1,p2),PremiumMin),PremiumMax))
            
            c0 = self.Copay0 + self.Sex*self.CopaySex + self.CopayAge*Age + self.CopayAgeSq*Age**2 + self.CopayInc*y + self.CopayIncSq*y**2 + self.CopayIncCu*y**3
            c1 = self.CopayHealth + self.CopayHealthAge*Age + self.CopayHealthAgeSq*Age**2 + self.CopayHealthInc*y + self.CopayHealthIncSq*y**2
            c2 = self.CopayHealthSq + self.CopayHealthSqAge*Age + self.CopayHealthSqAgeSq*Age**2 + self.CopayHealthSqInc*y + self.CopayHealthSqIncSq*y**2
            CopayFunc.append(LowerEnvelope(UpperEnvelope(QuadraticFunction(c0,c1,c2),CopayMin),CopayMax))
            
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
            bLvlGrid.append(makeGridExpMult(ming=self.IncomeNow[t], maxg=self.aXtraMax*self.IncomeNow[t], ng=self.bNrmCount+1, timestonest=self.aXtraNestFac))
        self.bLvlGrid = bLvlGrid
        
        self.addToTimeInv('bNrmGrid','hLvlGrid','Hgrid','Hcount')
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
        Constructs the time-invariant attributes CRRAmed and bFromxFunc.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        None
        '''
        rho = self.CRRA
        nu = rho*self.MedCurve
        self.bFromxFunc = TransConShareFunc(rho,nu)
        self.CRRAmed = nu
        self.addToTimeInv('CRRAmed','bFromxFunc')
        
        
    def updateHealthProdFuncs(self):
        '''
        Defines the time-invariant attributes HealthProdFunc, HealthProdInvFunc,
        MargHealthProdFunc, and MargHealthProdInvFunc.
        '''
        # Define the (marginal )(inverse) health production function
        self.HealthProdFunc = lambda i : self.HealthProd1*((i+self.HealthProd2)**self.HealthProd0 - self.HealthProd2**self.HealthProd0)
        self.MargHealthProdFunc = lambda i : self.HealthProd0*self.HealthProd1*(i+self.HealthProd2)**(self.HealthProd0-1.)
        self.HealthProdInvFunc = lambda x : (x/self.HealthProd1 + self.HealthProd2**self.HealthProd0)**(1./self.HealthProd0) - self.HealthProd2
        self.MargHealthProdInvFunc = lambda x : (x/(self.HealthProd0*self.HealthProd1))**(1./(self.HealthProd0-1.)) - self.HealthProd2
        self.addToTimeInv('HealthProdFunc','HealthProdInvFunc','MargHealthProdFunc','MargHealthProdInvFunc')
        
        
    def updateSubsidyFunc(self):
        '''
        Defines the time-invariant attribute SubsidyFunc.  This version makes a
        linear function of health using primitive parameters Subsidy0 and Subsidy1.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        None
        '''
        self.SubsidyFunc = LinearInterp([0.0, 1.0], [self.Subsidy0, self.Subsidy0 + self.Subsidy1], lower_extrap=True)
        self.addToTimeInv('SubsidyFunc')
        
        
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
        self.ConvexityFixer = JDfixer(self.aXtraGrid.size+16,self.Hgrid.size,self.MedShkCount,
                                      self.bNrmGrid.size,self.hLvlGrid.size,self.MedShkCount*3+1)
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
        
        
    def makeVaryingMedPrice(self,MedPriceHistory,t0):
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
        self.updateSubsidyFunc()
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
            setattr(self,var_name,copy(blank_array))
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
        MedPrice = self.MedPrice[t]
        
        bLvlNow = self.bLvlNow[these]
        hLvlNow = self.hLvlNow[these]
        MedShkNow = self.MedShkNow[these]
        cLvlNow, MedLvlNow, iLvlNow, xLvlNow = self.solution[t].PolicyFunc(bLvlNow,hLvlNow,MedShkNow)
        iLvlNow = np.maximum(iLvlNow,0.)
        
        MedShkBase = self.MedShkBase[these]
        Cfloor_better = MedShkBase > self.solution[t].CritDevFunc(bLvlNow,hLvlNow)
        cLvlNow[Cfloor_better] = self.Cfloor
        iLvlNow[Cfloor_better] = 0.0
        MedLvlNow[Cfloor_better] = (MedPrice*CopayNow[Cfloor_better])**(-1./self.CRRAmed)*MedShkNow[Cfloor_better]**(1.-1./self.CRRAmed)*self.Cfloor**(self.CRRA/self.CRRAmed)
        xLvlNow[Cfloor_better] = self.Cfloor + MedPrice*CopayNow[Cfloor_better]*MedLvlNow[Cfloor_better]
        
        SubsidyMax = self.SubsidyFunc(hLvlNow)
        iCostFull = MedPrice*iLvlNow
        iCostNow = CopayNow*np.maximum(iCostFull - SubsidyMax,0.0)
        SubsidyNow = np.minimum(iCostFull,SubsidyMax)
        OOPmedNow = MedPrice*MedLvlNow*CopayNow + iCostNow
        OOPmedNow[Cfloor_better] = np.maximum(bLvlNow[Cfloor_better] - PremiumNow[Cfloor_better] - self.Cfloor, 0.0)
        
        if ~hasattr(self,'cLvlNow'):
            self.cLvlNow = np.zeros(self.AgentCount)
            self.MedLvlNow = np.zeros(self.AgentCount)
            self.iLvlNow = np.zeros(self.AgentCount)
            self.xLvlNow = np.zeros(self.AgentCount)
            self.PremiumNow = np.zeros(self.AgentCount)
            self.CopayNow = np.zeros(self.AgentCount)
            self.OOPmedNow = np.zeros(self.AgentCount)
            self.SubsidyNow = np.zeros(self.AgentCount)
            
        self.PremiumNow[these] = PremiumNow
        self.CopayNow[these] = CopayNow
        self.cLvlNow[these] = cLvlNow
        self.MedLvlNow[these] = MedLvlNow
        self.iLvlNow[these] = iLvlNow
        self.xLvlNow[these] = xLvlNow
        self.OOPmedNow[these] = OOPmedNow
        self.SubsidyNow[these] = SubsidyNow
        self.PremiumNow[not_these] = np.nan
        self.CopayNow[not_these] = np.nan
        self.cLvlNow[not_these] = np.nan
        self.MedLvlNow[not_these] = np.nan
        self.iLvlNow[not_these] = np.nan
        self.xLvlNow[not_these] = np.nan
        self.OOPmedNow[not_these] = np.nan
        self.SubsidyNow[not_these] = np.nan
        
                
    def getPostStates(self):
        '''
        Calculates post states aLvlNow and HlvlNow.  Also optionally calculates some
        "socially optimal" values.
        '''
        t = self.t_sim
        aLvlNow = self.bLvlNow - self.PremiumNow - self.cLvlNow - self.OOPmedNow
        aLvlNow = np.maximum(aLvlNow,0.0) # Fixes those who go negative due to Cfloor help
        HlvlNow = self.ExpHealthNextFunc[t](self.hLvlNow) + self.HealthProdFunc(self.iLvlNow)
        RatioNow = np.maximum(self.MedPrice[t]*self.CopayNow*self.solution[t].dvdaFunc(aLvlNow,HlvlNow)/self.solution[t].dvdHfunc(aLvlNow,HlvlNow),0.0)
        
        self.aLvlNow = aLvlNow
        self.HlvlNow = HlvlNow
        self.RatioNow = RatioNow
        self.TotalMedNow = self.MedPrice[t]*(self.MedLvlNow + self.iLvlNow)
        
        if self.CalcSocialOptimum and self.CalcExpectationFuncs: # This is False during solution and most counterfactuals
            # Extract and rename needed objects for easy reference
            fp_inv = self.MargHealthProdInvFunc
            fp   = self.MargHealthProdFunc
            p_L  = self.LifePrice   # "value" of a year of life
            pi   = self.MedPrice[t] # price of unit of care
            iLvl = self.iLvlNow     # actual health investment this period
            dvda = self.solution[t].dvdaFunc(self.aLvlNow,self.HlvlNow)
            dvdH = self.solution[t].dvdHfunc(self.aLvlNow,self.HlvlNow)
            dLdH = self.solution[t].ExpectedLifeFunc.derivativeY(self.aLvlNow,self.HlvlNow) # marginal life expectancy from post-health
            dOdH = self.solution[t].TotalMedPDVfunc.derivativeY(self.aLvlNow,self.HlvlNow) # marginal PDV lifetime medical expenses from post-health
            
            # Calculate "socially optimal" values
            self.iLvlSocOpt = fp_inv(pi/(p_L*dLdH - dOdH)) # "socially optimal" health investment
            self.LifePriceSocOpt = pi/(fp(iLvl)*dLdH) + dOdH/dLdH # "value" of life year that *would* make actual health investment "socially optimal"
            self.CopaySocOpt = dvdH/(dvda*(p_L*dLdH - dOdH)) # coinsurance rate that *would have* made agent choose "socially optimal" health investment  

    
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

        
    def plotdvdbFuncByHealth(self,t,bMin=None,bMax=20.0,hSet=None,pseudo_inverse=False,Alt=False):
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
            elif Alt:
                dvdb = self.solution[t].vFunc.derivativeX(B,hLvl*some_ones)
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
        
        
    def plot2DfuncByHealth(self,func_name,t,bMin=None,bMax=20.0,hSet=None):
        '''
        Plot a function defined on bLvl x hLvl at a set of health values.
        
        Parameters
        ----------
        func_name : str
            Name of the function to be plotted, which should be an attribute of self.solution
        t : int
            Period of life to plot the function.
        bMin : float or None
            Minimum value of bLvl at which to begin plot.  Defaults to lowest
            valid value of bLvl.
        bMax : float
            Highest value of bLvl at which to end plot.
        hSet : [float]
            List of hLvl values at which to plot the function.  Defaults to full set of
            hLvl in the interpolated function.
        
        Returns
        -------
        None
        '''
        func = getattr(self.solution[t],func_name)
        if hSet is None:
            hSet = func.y_list
        if bMin is None:
            bMin = func.x_list[0]
            
        B = np.linspace(bMin,bMax,300)
        some_ones = np.ones_like(B)
        for hLvl in hSet:
            M = func(B,hLvl*some_ones)
            plt.plot(B,M)
        plt.xlabel('Market resources bLvl')
        plt.ylabel(func_name)
        plt.show()
        
    
    def plot2DfuncByWealth(self,func_name,t,hMin=0.0,hMax=1.0,bSet=None):
        '''
        Plot a function defined on bLvl x hLvl at a set of health values.
        
        Parameters
        ----------
        func_name : str
            Name of the function to be plotted, which should be an attribute of self.solution
        t : int
            Period of life to plot the function.
        hMin : float
            Minimum value of hLvl at which to begin plot.  Defaults to 0.0
        hMax : float
            Highest value of hLvl at which to end plot.  Defaults to 1.0
        bSet : [float]
            List of bLvl values at which to plot the function.  Defaults to full set of
            bLvl in the interpolated function.
        
        Returns
        -------
        None
        '''
        func = getattr(self.solution[t],func_name)
        if bSet is None:
            bSet = func.x_list
            
        H = np.linspace(hMin,hMax,300)
        some_ones = np.ones_like(H)
        for bLvl in bSet:
            M = func(bLvl*some_ones,H)
            plt.plot(H,M)
        plt.xlabel('Health hLvl')
        plt.ylabel(func_name)
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

    
