'''
This module contains the solver and agent class for the model for "...An Ounce of Prevention..."
'''

import sys
import os
sys.path.insert(0,'../')
sys.path.insert(0,'../ConsumptionSaving/')

import numpy as np
from HARKcore import NullFunc
from HARKinterpolation import ConstantFunction
from HARKutilities import makeGridExpMult
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
    
    def inverse(self,x):
        a = self.a0
        b = self.a1
        c = self.a2 - x
        discrim_arg = b**2 - 4.*a*c
        discrim_arg(discrim_arg < 0.) = np.nan
        discrim = np.sqrt(discrim_arg)
        if a < 0.:
            out = (-b - discrim)/(2.*a)
        elif a > 0.:
            out = (-b + discrim)/(2.*a)
        else:
            out = -c/b
        return out
    
    
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
        if dvdbfunc is None:
            dvdb = NullFunc()
        if dvdhfunc is None:
            dvdh = NullFunc()
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
                optMedZeroFunc = lambda q : 1. + CRRAmed/(MedPrice*MedShk)*xLvl/(1.+q) - np.exp((xLvl/MedPrice*q/(1.+q))/MedShk)
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
            theta0 = self.Mortality0 + self.MortalityAge*t + self.MortalityAge*t**2
            theta1 = self.MortalityHealth
            theta2 = self.MortalityHealthSq
            LivPrbFunc.append(QuadraticFunction(theta0,theta1,theta2))
            
            gamma0 = self.HealthNext0 + self.HealthNextAge*t + self.HealthNextAgeSq*t**2
            gamma1 = self.HealthNextHealth
            gamma2 = self.HealthNextHealthSq
            ThisHealthFunc = QuadraticFunction(gamma0,gamma1,gamma2)
            ExpHealthNextFunc.append(ThisHealthFunc)
            ExpHealthNextInvFunc.append(ThisHealthFunc.inverse)
        
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
            y = self.Income[t]
            p0 = self.Premium0 + self.PremiumAge*t + self.PremiumAge*t**2 + self.PremiumInc*y + self.PremiumIncSq*y**2 + self.PremiumIncCu*y**3
            p1 = self.PremiumHealth + self.PremiumHealthAge*t + self.PremiumHealthAgeSq*t**2 + self.PremiumHealthInc*y + self.PremiumHealthIncSq*y**2
            p2 = self.PremiumHealthSq + self.PremiumHealthSqAge*t + self.PremiumHealthSqAgeSq*t**2 + self.PremiumHealthSqInc*y + self.PremiumHealthSqIncSq*y**2
            PremiumFunc.append(QuadraticFunction(p0,p1,p2))
            
            c0 = self.Copay0 + self.CopayAge*t + self.CopayAge*t**2 + self.CopayInc*y + self.CopayIncSq*y**2 + self.CopayIncCu*y**3
            c1 = self.CopayHealth + self.CopayHealthAge*t + self.CopayHealthAgeSq*t**2 + self.CopayHealthInc*y + self.CopayHealthIncSq*y**2
            c2 = self.CopayHealthSq + self.CopayHealthSqAge*t + self.CopayHealthSqAgeSq*t**2 + self.CopayHealthSqInc*y + self.CopayHealthSqIncSq*y**2
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
        
        hGrid = np.linspace(0.,1.,self.hCount)
        Hgrid = np.linspace(0.,1.,self.Hcount)
        
        self.bNrmGrid = bNrmGrid
        self.hGrid = hGrid
        self.Hgrid = Hgrid
        self.addToTimeInv('bNrmGrid','hGrid','Hgrid')
        
        
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
        xLvlGrid = makeGridExpMult(self.aXtraMin,self.aXtraMax,self.aXtraCount*xLvl_aug_factor,self.aXtraNestFac)
        
        # Make a master grid of medical need shocks
        ShkLogMin = self.MedShkMeanFunc[t0](1.0) - 3.0*self.MedShkStdFunc[t0](1.0)
        ShkLogMax = self.MedShkMeanFunc[T](0.0) + 5.0*self.MedShkStdFunc[T](0.0)
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
        MedGrid = qGrid/(1.+qGrid)*xGrid/MedPriceEff
        cEffGrid = (1.-np.exp(-MedGrid/ShkGrid))**self.MedCurve*cGrid
        cEffGrid[:,0] = xVec

        # Construct a grid of marginal effective consumption values
        temp1 = np.exp(-MedGrid/ShkGrid)
        temp2 = np.exp(MedGrid/ShkGrid)
        dcdx = temp2/(temp2 + self.MedCurve)
        dcdx[ShkZero] = 1.
        dMeddx = (1./MedPriceEff)*self.MedCurve/(temp2 + self.MedCurve)
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
        DfuncList.append(LogOnLogFunc2D(LinearInterpOnInterp1D(DfuncBase_by_Shk_list,ShkVec)))
        DpFuncList.append(MargCostFunc(LinearInterpOnInterp1D(DpFuncBase_by_Shk_list,ShkVec)))
        GfuncList.append(LogOnLogFunc2D(LinearInterpOnInterp1D(GfuncBase_by_Shk_list,ShkVec)))
        
        # Make a 2D interpolation for the cEffFunc
        LogcEffGrid = np.log(cEffGrid[1:,:])
        LogxVec = np.log(xVec[1:])
        cEffFunc = LogOnLogFunc2D(BilinearInterp(LogcEffGrid,LogxVec,ShkVec))
        cEffFuncList.append(cEffFunc)
        
        
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
        self.pseudo_terminal = True
        
        