'''
A model of consumption, savings, medical spending, and medical insurance selection.
'''
import sys 
sys.path.insert(0,'../../')
sys.path.insert(0,'../../ConsumptionSaving')
import numpy as np
from copy import copy, deepcopy
from time import clock

from HARKcore import HARKobject, AgentType
from HARKutilities import CRRAutility, CRRAutilityP, CRRAutilityPP, CRRAutilityP_inv,\
                          CRRAutility_invP, CRRAutility_inv, CRRAutilityP_invP, NullFunc,\
                          approxLognormal, addDiscreteOutcome
from HARKinterpolation import LinearInterp, CubicInterp, BilinearInterpOnInterp1D, LinearInterpOnInterp1D, \
                              LinearInterpOnInterp2D,UpperEnvelope, TrilinearInterp, ConstantFunction, CompositeFunc2D, \
                              VariableLowerBoundFunc2D, VariableLowerBoundFunc3D, VariableLowerBoundFunc3Dalt, \
                              BilinearInterp, CompositeFunc3D, IdentityFunction
from HARKsimulation import drawUniform, drawLognormal, drawMeanOneLognormal, drawDiscrete, drawBernoulli
from ConsMedModel import MedShockPolicyFunc, MedShockConsumerType
from ConsIndShockModel import ValueFunc
from ConsPersistentShockModel import ValueFunc2D, MargValueFunc2D, MargMargValueFunc2D
from ConsMarkovModel import MarkovConsumerType
from ConsIndShockModel import constructLognormalIncomeProcessUnemployment
from JorgensenDruedahl import makeGridDenser, JDfixer
import matplotlib.pyplot as plt
from scipy.stats import norm
                                     
utility       = CRRAutility
utilityP      = CRRAutilityP
utilityPP     = CRRAutilityPP
utilityP_inv  = CRRAutilityP_inv
utility_invP  = CRRAutility_invP
utility_inv   = CRRAutility_inv
utilityP_invP = CRRAutilityP_invP

class LogFunc1D(HARKobject):
    '''
    A trivial class for representing R --> R functions whose output has been
    transformed through the log function (must be exp'ed to get true level).
    '''
    distiance_criteria = ['func']
    
    def __init__(self,func):
        self.func = func
        
    def __call__(self,x):
        return np.exp(self.func(x))
    

class TwistFuncA(HARKobject):
    '''
    A trivial class for representing R^3 --> R functions whose inputs must be
    permuted from (A,B,C) to (C,A,B).
    '''
    distiance_criteria = ['func']
    
    def __init__(self,func):
        self.func = func
        
    def __call__(self,x,y,z):
        return self.func(z,x,y)
    
    
class TwistFuncB(HARKobject):
    '''
    A trivial class for representing R^3 --> R functions whose inputs must be
    permuted from (A,B,C) to (C,A,B).
    '''
    distiance_criteria = ['func']
    
    def __init__(self,func):
        self.func = func
        
    def __call__(self,x,y,z):
        return self.func(x,z,y)
    
    def derivativeX(self,x,y,z):
        return self.func.derivativeX(x,z,y)
    
    def derivativeY(self,x,y,z):
        return self.func.derivativeY(x,z,y)
    
    def derivativeZ(self,x,y,z):
        return self.func.derivativeZ(x,z,y)
    

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


class InsuranceSelectionSolution(HARKobject):
    '''
    Class for representing the single period solution of the insurance selection model.
    '''
    distance_criteria = ['vPfunc']
    
    def __init__(self, policyFunc=None, vFunc=None, vFuncByContract=None, vPfunc=None, vPPfunc=None,
                 AVfunc=None, mLvlMin=None, hLvl=None):
        '''
        The constructor for a new InsuranceSelectionSolution object.
        
        Parameters
        ----------
        policyFunc : [[InsSelPolicyFunc]]
            The policy functions for this period.  policyFunc[h][j] represents the policy function
            when holding contract j in health state h (after paying premiums).
        vFunc : [ValueFunc2D]
            The beginning-of-period value functions for this period, defined over
            market resources: v = vFunc(mLvl,pLvl); list corresponds to health states.
        vFuncByContract : [[ValueFunc2D]]
            Value function for holding a contract in a particular health state.  vFuncByContract[h][j]
            is value for contract j in health state h.
        vPfunc : [MargValueFunc2D]
            The beginning-of-period marginal value functions for this period,
            defined over market resources: vP = vPfunc(mLvl,pLvl); list corresponds to health states.
        vPPfunc : [MargMargValueFunc2D]
            The beginning-of-period marginal marginal value functions for this
            period, defined over market resources: vPP = vPPfunc(mLvl,pLvl); list is for health.
        AVfunc : HARKinterpolator2D
            Actuarial value as a function of permanent income and market resources, by contract.
        mLvlMin : function
            The minimum allowable market resources for this period; the consump-
            tion function (etc) are undefined for m < mLvlMin.  Does not vary with health.
        hLvl : [function]
            Human wealth after receiving income this period: PDV of all future
            income, ignoring mortality.  List corresponds to health states.
            
        Returns
        -------
        None        
        '''
        # Change any missing function inputs to NullFunc or empty sets
        if policyFunc is None:
            policyFunc = []
        if vFunc is None:
            vFunc = []
        if vFuncByContract is None:
            vFuncByContract = []
        if vPfunc is None:
            vPfunc = []
        if vPPfunc is None:
            vPPfunc = []
        if AVfunc is None:
            AVfunc = []
        if hLvl is None:
            hLvl = []
        if mLvlMin is None:
            mLvlMin = NullFunc
            
        self.policyFunc   = copy(policyFunc)
        self.vFunc        = copy(vFunc)
        self.vFuncByContract = copy(vFuncByContract)
        self.vPfunc       = copy(vPfunc)
        self.vPPfunc      = copy(vPPfunc)
        self.AVfunc       = copy(AVfunc)
        self.mLvlMin      = copy(mLvlMin)
        self.hLvl         = copy(hLvl)
        
    def getMemSize(self):
        temp = 0
        for h in range(len(self.vFunc)):
            temp += self.vFunc[h].getMemSize()
            temp += self.vPfunc[h].getMemSize()
            for z in range(len(self.vFuncByContract[h])):
                temp += self.vFuncByContract[h][z].getMemSize()
                temp += self.policyFunc[h][z].getMemSize()
        temp += self.mLvlMin.getMemSize()
        return temp
        
        
    def appendSolution(self,policyFunc=None, vFunc=None, vFuncByContract=None, vPfunc=None, vPPfunc=None, AVfunc=None, hLvl=None):
        '''
        Add data for a new discrete health state to an existion solution object.
        
        Parameters
        ----------
        policyFunc : [InsSelPolicyFunc]
            The policy functions for this period in this health state.  policyFunc[j] represents the
            policy function when holding contract j (after paying premiums).
        vFunc : ValueFunc2D
            The beginning-of-period value functions for this period in this health state, defined
            overmarket resources: v = vFunc(mLvl,pLvl).
        vFuncByContract : [ValueFunc2D]
            Value function for holding each contract.  vFuncByContract[j] is value for contract j.
        vPfunc : MargValueFunc2D
            The beginning-of-period marginal value functions for this period in this health state,
            defined over market resources: vP = vPfunc(mLvl,pLvl); list corresponds to health states.
        vPPfunc : MargMargValueFunc2D
            The beginning-of-period marginal marginal value functions for this period in this health
            state, defined over market resources: vPP = vPPfunc(mLvl,pLvl).
        AVfunc : HARKinterpolator2D
            Actuarial value as a function of permanent income and market resources, by contract.
        hLvl : function
            Human wealth after receiving income this period in this health state: PDV of all future
            income, ignoring mortality.
            
        Returns
        -------
        None        
        '''
        self.policyFunc.append(policyFunc)
        self.vFunc.append(vFunc)
        self.vFuncByContract.append(vFuncByContract)
        self.vPfunc.append(vPfunc)
        self.vPPfunc.append(vPPfunc)
        self.AVfunc.append(AVfunc)
        self.hLvl.append(hLvl)


class ValueFunc3D(HARKobject):
    '''
    The value function for a particular coinsurance rate, defined over market resources
    (after paying premiums and "option cost"), permanent income, and the medical shock.
    '''
    distance_criteria = ['vFuncNvrs','CRRA']
    
    def __init__(self,vFuncNvrs,CRRA):
        '''
        Make a new instance of ValueFunc3D.
        
        Parameters
        ----------
        vFuncNvrs : function
            Pseudo-inverse value function for this coinsurance rate, defined over market resources,
            permanent income, and medical shock.  Usually a HARKinterpolator3D.
        CRRA : float
            Coefficient of relative risk aversion for re-curving the pseudo-inverse value function.
            
        Returns
        -------
        None
        '''
        #self.vFuncNvrs = deepcopy(vFuncNvrs)
        self.vFuncNvrs = vFuncNvrs
        self.CRRA = CRRA
        
    def getMemSize(self):
        return self.vFuncNvrs.getMemSize()
        
    def __call__(self,mLvl,pLvl,MedShk):
        '''
        Evaluate the value function for this coinsurance rate at the requested states.
        
        Parameters
        ----------
        mLvl : np.array
             Market resource levels.
        pLvl : np.array
             Permanent income levels.
        MedShk : np.array
             Medical need shocks.
             
        Returns
        -------
        v : np.array
            Value at each input state.
        '''
        vNvrs = self.vFuncNvrs(mLvl,pLvl,MedShk)
        v = utility(vNvrs,self.CRRA)
        return v


class InsSelPolicyFunc(HARKobject):
    '''
    The policy function for a particular medical insurance contract, defined over market resources
    (after paying premiums), permanent income, and the medical needs shock.
    '''
    distance_criteria = ['ValueFuncFullPrice','ValueFuncCopay','PolicyFuncFullPrice','PolicyFuncCopay','OptionCost']
    
    def __init__(self,ValueFuncFullPrice,ValueFuncCopay,PolicyFuncFullPrice,PolicyFuncCopay,DpFuncFullPrice,DpFuncCopay,Contract,CRRAmed):
        '''
        Make a new instance of ValueFuncContract.
        
        Parameters
        ----------
        ValueFuncFullPrice : ValueFunc3D
            Value function (over market resources after premiums / "option cost", permanent income,
            and medical shock) when the individual pays full price for care.
        ValueFuncCopay : ValueFunc3D
            Value function (over market resources after premiums / "option cost", permanent income,
            and medical shock) when the individual pays the coinsurance rate for care.
        PolicyFuncFullPrice : MedShockPolicyFunc
            Policy function when paying full price for care, including consumption and medical care,
            defined over market resources (after option cost), permanent income, and medical shock.
        PolicyFuncFullPrice : MedShockPolicyFunc
            Policy function when paying the coinsurance rate for care, including consumption and
            medical care, defined over market resources, permanent income, and medical shock.
        DpFuncFullPrice : function
            Function that gives marginal cost of achieving a given effective consumption level
            when facing a given medical shock, when paying full price for care.
        DpFuncFullCopay : function
            Function that gives marginal cost of achieving a given effective consumption level
            when facing a given medical shock, when paying the coinsurance rate for care.
        Contract : MedInsuranceContract
            Medical insurance contract of interest.
        CRRAmed : float
            Curvature parameter nu for medical care in effective consumption calculation.
            
        Returns
        -------
        None
        '''
        self.ValueFuncFullPrice = ValueFuncFullPrice
        self.ValueFuncCopay = ValueFuncCopay
        self.PolicyFuncFullPrice = PolicyFuncFullPrice
        self.PolicyFuncCopay = PolicyFuncCopay
        self.DpFuncFullPrice = DpFuncFullPrice
        self.DpFuncCopay = DpFuncCopay
        self.Contract = Contract
        self.OptionCost = Contract.Deductible*(1.0-Contract.Copay)
        self.CRRAmed = CRRAmed
               
    def __call__(self,mLvl,pLvl,MedShk):
        '''
        Evaluate the policy function for this contract.
        
        Parameters
        ----------
        mLvl : np.array
             Market resource levels.
        pLvl : np.array
             Permanent income levels.
        MedShk : np.array
             Medical need shocks.
             
        Returns
        -------
        cLvl : np.array
            Optimal consumption at each point in the input.
        MedLvl : np.array
            Optimal medical care at each point in the input.
        '''
        if mLvl.shape is ():
            mLvl = np.array([mLvl])
            pLvl = np.array([pLvl])
            MedShk = np.array([MedShk])
            float_in = True
        else:
            float_in = False
        
        # Get value of paying full price or paying "option cost" to pay coinsurance rate
        mTemp = mLvl-self.OptionCost
        v_Copay = self.ValueFuncCopay(mTemp,pLvl,MedShk)
        v_Copay[mTemp < 0.] = -np.inf
        if self.OptionCost > 0.:
            v_FullPrice = self.ValueFuncFullPrice(mLvl,pLvl,MedShk)
        
        # Decide which option is better and initialize output
        if self.OptionCost > 0.:
            Copay_better = v_Copay > v_FullPrice
        else:
            Copay_better = np.ones_like(v_Copay,dtype=bool)
        FullPrice_better = np.logical_not(Copay_better)
        cLvl = np.zeros_like(mTemp)
        MedLvl = np.zeros_like(mTemp)
        
        # Fill in output using better of two choices
        cLvl[Copay_better], MedLvl[Copay_better] = self.PolicyFuncCopay(mTemp[Copay_better],pLvl[Copay_better],MedShk[Copay_better])
        cLvl[FullPrice_better], MedLvl[FullPrice_better] = self.PolicyFuncFullPrice(mLvl[FullPrice_better],pLvl[FullPrice_better],MedShk[FullPrice_better])
        MedLvl[MedShk == 0.0] = 0.0
        
        if float_in:
            return cLvl[0], MedLvl[0]
        else:
            return cLvl, MedLvl
        
    def xFunc(self,mLvl,pLvl,MedShk,return_copay_bool=False):
        '''
        Evaluate the expenditure function for this contract.
        
        Parameters
        ----------
        mLvl : np.array
             Market resource levels.
        pLvl : np.array
             Permanent income levels.
        MedShk : np.array
             Medical need shocks.
        return_copay_bool : bool
            Indicator for whether to return the boolean array Copay_better.
             
        Returns
        -------
        xLvl : np.array
            Optimal expenditure at each point in the input.
        '''
        if mLvl.shape is ():
            mLvl = np.array([mLvl])
            pLvl = np.array([pLvl])
            MedShk = np.array([MedShk])
            float_in = True
        else:
            float_in = False
        
        # Get value of paying full price or paying "option cost" to pay coinsurance rate
        mTemp = mLvl-self.OptionCost
        v_Copay = self.ValueFuncCopay(mTemp,pLvl,MedShk)
        v_Copay[mTemp < 0.] = -np.inf
        if self.OptionCost > 0.:
            v_FullPrice = self.ValueFuncFullPrice(mLvl,pLvl,MedShk)
        
        # Decide which option is better and initialize output
        if self.OptionCost > 0.:
            Copay_better = v_Copay > v_FullPrice
        else:
            Copay_better = np.ones_like(v_Copay,dtype=bool)
        FullPrice_better = np.logical_not(Copay_better)
        xLvl = np.zeros_like(mTemp)
        
        # Fill in output using better of two choices
        xLvl[Copay_better] = self.PolicyFuncCopay.xFunc(mTemp[Copay_better],pLvl[Copay_better],MedShk[Copay_better])
        xLvl[FullPrice_better] = self.PolicyFuncFullPrice.xFunc(mLvl[FullPrice_better],pLvl[FullPrice_better],MedShk[FullPrice_better])
        
        if float_in:
            return xLvl[0]
        else:
            if return_copay_bool: # Return Copay_better if requested
                return xLvl, Copay_better
            else:
                return xLvl
        

    def evalvAndvPandvPP(self,mLvl,pLvl,MedShk):
        '''
        Evaluate the value, marginal value, and marginal marginal value for this contract.
        
        Parameters
        ----------
        mLvl : np.array
             Market resource levels.
        pLvl : np.array
             Permanent income levels.
        MedShk : np.array
             Medical need shocks.
             
        Returns
        -------
        cLvl : np.array
            Optimal consumption at each point in the input.
        MedLvl : np.array
            Optimal medical care at each point in the input.
        v : np.array
            Value at each point in the input.
        vP : np.array
            Marginal value (with respect to market resources) at each point in the input.
        vPP : np.array
            Marginal marginal value (with respect to market resources) at each point in the input.
        '''
        # Get value of paying full price or paying "option cost" to pay coinsurance rate
        mTemp = mLvl-self.OptionCost
        v_FullPrice = self.ValueFuncFullPrice(mLvl,pLvl,MedShk)
        v_Copay     = self.ValueFuncCopay(mTemp,pLvl,MedShk)
        v_Copay[mTemp < 0.] = -np.inf
        
        # Decide which option is better and initialize output
        Copay_better = v_Copay > v_FullPrice
        FullPrice_better = np.logical_not(Copay_better)
        cLvl = np.zeros_like(mLvl)
        MedLvl = np.zeros_like(mLvl)
        
        # Fill in output using better of two choices
        cLvl[Copay_better], MedLvl[Copay_better] = self.PolicyFuncCopay(mTemp[Copay_better],pLvl[Copay_better],MedShk[Copay_better])
        cLvl[FullPrice_better], MedLvl[FullPrice_better] = self.PolicyFuncFullPrice(mLvl[FullPrice_better],pLvl[FullPrice_better],MedShk[FullPrice_better])

        # Find the marginal cost of achieving this effective consumption level
        ShkZero = MedShk==0.
        cEffLvl = (1.-np.exp(-MedLvl/MedShk))**self.CRRAmed*cLvl
        cEffLvl[ShkZero] = cLvl[ShkZero]
        Dp = np.zeros_like(cEffLvl)
        Dp[Copay_better] = self.DpFuncCopay(cEffLvl[Copay_better],MedShk[Copay_better])
        Dp[FullPrice_better] = self.DpFuncFullPrice(cEffLvl[FullPrice_better],MedShk[FullPrice_better])
        
        # Calculate value and marginal value and return them
        v   = np.maximum(v_Copay,v_FullPrice)
        vP  = utilityP(cEffLvl,self.ValueFuncFullPrice.CRRA)/Dp
        vPP = np.zeros_like(vP) # Don't waste time calculating
        return v, vP, vPP
        
        
class MedShockPolicyFuncPrecalc(MedShockPolicyFunc):
    '''
    Class for representing the policy function in the medical shocks model: opt-
    imal consumption and medical care for given market resources, permanent income,
    and medical need shock.  Always obeys Con + MedPrice*Med = optimal spending.
    Replaces initialization method of parent class, as the bFromxFunc is passed as
    an argument to the constructor rather than assembled by it.
    '''
    def __init__(self,xFunc,bFromxFunc,MedPrice):
        '''
        Make a new MedShockPolicyFuncPrecalc.
        
        Parameters
        ----------
        xFunc : function
            Optimal total spending as a function of market resources, permanent
            income, and the medical need shock.
        bFromxFunc : function
            Optimal consumption ratio as a function of total expenditure and the
            medical need shock.
        MedPrice : float
            Relative price of a unit of medical care.
        
        Returns
        -------
        None
        '''
        self.xFunc = xFunc
        self.bFunc = bFromxFunc
        self.MedPrice = MedPrice # This should be the "effective" price (accounting for copay)
      
      
class MedInsuranceContract(HARKobject):
    '''
    A medical insurance contract characterized by a premium (as a function of market resources and
    permanent income), a deductible, and a coinsurance rate. Implicitly conditional on health.
    '''
    distance_criteria = []
    
    def __init__(self,Premium,Deductible,Copay,MedPrice):
        '''
        Make a new insurance constract.
        
        Parameters
        ----------
        Premium : function
            Net premium for this contract as a function of market resources and permanent income.
        Deductible : float
            Deductible for this contract.  Below the deductible, policyholder pays full price for care.
            Above the deductible, policyholder pays coinsurance rate.
        Copay : float
            Coinsurance rate for the contract (above deductible).
        MedPrice : float
            Relative cost of care for the period in which this contract is sold.
        '''
        self.Premium = Premium
        self.Deductible = Deductible
        self.Copay = Copay
        self.MedPrice = MedPrice
        
        # Define an out-of-pocket spending function that maps quantity of care purchased to OOP cost
        kink_point = Deductible/MedPrice
        if kink_point > 0.0:
            OOPfunc = LinearInterp(np.array([0.0,kink_point,kink_point+1.0]),np.array([0.0,Deductible,Deductible+MedPrice*Copay]))
        else: # Fixes a problem with Deductible = 0, which would create an invalid x_list for interpolation
            OOPfunc = LinearInterp(np.array([0.0,1.0]),np.array([0.0,MedPrice*Copay]))
        self.OOPfunc = OOPfunc
        
    def getMemSize(self):
        return self.OOPfunc.getMemSize()
        
####################################################################################################

def solveInsuranceSelectionStatic(solution_next,MedShkDstn,CRRA,MedPrice,xLvlGrid,PolicyFuncList):
    '''
    Solves one period of the "static" version of the insurance selection model.
    In this model, total spending equals income, so there is no dependency on t+1.
    
    Parameters
    ----------
    solution_next : InsuranceSelectionSolution
        The solution to next period's one period problem.
    MedShkDstn : [[np.array]]
        Discrete distribution of the multiplicative utility shifter for med-
        ical care. Order: probabilities, preference shocks.  Distribution
        depends on discrete health state.
    CRRA : float
        Coefficient of relative risk aversion for composite consumption.
    MedPrice : float
        Price of unit of medical care relative to unit of consumption.
    xLvlGrid : np.array
        Grid of spending levels (after paying premiums).
    PolicyFuncList : [[InsSelPolicyFunc]]
        List of policy functions for each contract that could be purchased this
        period for each health state.
        
    Returns
    -------
    solution_now : InsuranceSelectionSolution
        Solution to this period's static problem.
    '''
    StateCount = len(MedShkDstn)
    x_N = xLvlGrid.size
    AVfuncs = []
    vFuncs = []
    
    # Loop through each discrete state
    for j in range(StateCount):
        # Unpack the shock distribution and initialize objects
        MedShkDstnNow = MedShkDstn[j]
        MedShkVals = MedShkDstnNow[1]
        MedShkPrbs = MedShkDstnNow[0]
        Shk_N = MedShkVals.size
        xLvlGrid_tiled = np.tile(np.reshape(xLvlGrid,(x_N,1)),(1,Shk_N))
        MedShkVals_tiled = np.tile(np.reshape(MedShkVals,(1,Shk_N)),(x_N,1))
        ones_tiled = np.ones_like(xLvlGrid_tiled)
        
        # Initialize the list of solutions
        AVfuncs_this_state = []
        vFuncs_this_state = []

        # Loop through the various contracts
        for policyFunc in PolicyFuncList[j]:
            # Get medical expenses, (marginal) value, and insurance payout
            cLvl, MedLvl = policyFunc(xLvlGrid_tiled,ones_tiled,MedShkVals_tiled)
            cLvl[0,:] = 0.0
            MedLvl[0,:] = 0.0
            vGrid, vPgrid, trash = policyFunc.evalvAndvPandvPP(xLvlGrid_tiled,ones_tiled,MedShkVals_tiled)
            InsPayGrid = MedPrice*MedLvl - policyFunc.Contract.OOPfunc(MedLvl)
            
            # Calculate expectations across medical need shocks
            ActuarialValue = np.sum(InsPayGrid*MedShkPrbs,axis=1)
            vNow = np.sum(vGrid*MedShkPrbs,axis=1)
            vPnow = np.sum(vPgrid*MedShkPrbs,axis=1)
            
            # Construct solution functions for this contract-state
            AVfunc = LinearInterp(xLvlGrid,ActuarialValue)
            vNvrs = utility_inv(vNow,CRRA)
            vNvrs[0] = 0.0
            vNvrsP = vPnow*utility_invP(vNow,CRRA)
            vNvrsP[0] = 0.0
            #vNvrsFunc = CubicInterp(xLvlGrid,vNvrs,vNvrsP)
            vNvrsFunc = LinearInterp(xLvlGrid,vNvrs)
            vFunc = ValueFunc(vNvrsFunc,CRRA)
            
            # Add the functions to the list of solutions
            AVfuncs_this_state.append(AVfunc)
            vFuncs_this_state.append(vFunc)
        
        # Add this health state to the overall solution
        AVfuncs.append(AVfuncs_this_state)
        vFuncs.append(vFuncs_this_state)
    
    # Construct an object to return as the solution
    solution_now = InsuranceSelectionSolution()
    solution_now.policyFunc = PolicyFuncList
    solution_now.vFuncByContract = vFuncs
    solution_now.AVfunc = AVfuncs
    #print('Solved a period of the static model!')
    return solution_now


def solveInsuranceSelection(solution_next,IncomeDstn,MedShkDstn,MedShkAvg,MedShkStd,LivPrb,DiscFac,
                            CRRA,CRRAmed,Cfloor,Rfree,MedPrice,pLvlNextFunc,BoroCnstArt,aXtraGrid,
                            pLvlGrid,ContractList,MrkvArray,ChoiceShkMag,CopayList,bFromxFuncList,
                            DfuncList,DpFuncList,cEffFuncList,GfuncList,CritShkFuncList,CubicBool):
    '''
    Solves one period of the insurance selection model.
    
    Parameters
    ----------
    solution_next : InsuranceSelectionSolution
        The solution to next period's one period problem.
    IncomeDstn : [[np.array]]
        Lists containing three arrays of floats, representing a discrete
        approximation to the income process between the period being solved
        and the one immediately following (in solution_next). Order: event
        probabilities, permanent shocks, transitory shocks.  Distribution
        depends on discrete health state.
    MedShkDstn : [[np.array]]
        Discrete distribution of the multiplicative utility shifter for med-
        ical care. Order: probabilities, preference shocks.  Distribution
        depends on discrete health state.
    MedShkAvg : [float]
        Mean value of log medical need shocks by health state.
    MedShkStd : [float]
        Standard deviation of log medical need shocks by health state.
    LivPrb : np.array
        Survival probability; likelihood of being alive at the beginning of
        the succeeding period conditional on current health state.   
    DiscFac : float
        Intertemporal discount factor for future utility.        
    CRRA : float
        Coefficient of relative risk aversion for composite consumption.
    CRRAmed : float
        Effective consumption curvature parameter for medical care.
    Cfloor : float
        Floor on effective consumption, set by policy.
    Rfree : np.array
        Risk free interest factor on end-of-period assets conditional on
        next period's health state.  Actually constant across states.
    MedPrice : float
        Price of unit of medical care relative to unit of consumption.
    pLvlNextFunc : function
        Next period's expected permanent income level as a function of current pLvl.
    BoroCnstArt: float or None
        Borrowing constraint for the minimum allowable assets to end the
        period with.
    aXtraGrid: np.array
        Array of "extra" end-of-period (normalized) asset values-- assets
        above the absolute minimum acceptable level.
    pLvlGrid: np.array
        Array of permanent income levels at which to solve the problem.
    ContractList : [[MedInsuranceContract]]
        Lists of medical insurance contracts for each discrete health state (list of lists).
    MrkvArray : numpy.array
        An NxN array representing a Markov transition matrix between discrete
        health states conditional on survival.  The i,j-th element of MrkvArray
        is the probability of moving from state i in period t to state j in period t+1.
    ChoiceShkMag : float
        Magnitude of T1EV preference shocks for each insurance contract when making selection.
        Shocks are applied to pseudo-inverse value of contracts.
    CopayList : [float]
        Set of coinsurance rates across all contracts in ContractList, including 1.0 (full price).
    bFromxFuncList : [function]
        Optimal consumption ratio as a function of total expenditure and medical need shock by
        coinsurance rate.  Elements of this list correspond to coinsurance rates in CopayList.
    DfuncList : [function]
        List of cost functions for each copay rate that can be achieved this period.  Each
        function takes an "effective consumption" level and medical shock and returns the
        cost of achieving that cEffLvl.
    DpFuncList : [function]
        List of marginal cost functions for each copay rate that can be achieved this period.
        Each function takes an "effective consumption" level and medical shock and returns the
        marginal cost of a bit more effective consumption at that cEffLvl.
    cEffFuncList : [function]
        List of functions for each copay rate this period that give the maximum effective
        consumption as a function of total spending (on consumption and medical care) and
        the medical need shock.
    GfuncList : [function]
        List of functions for each copay that solve the first order condition for effective
        consumption.  Each function takes in a pseudo-inverse end-of-period marginal value
        and a medical shock and return the level of effective consumption that solves the
        first order condition for optimality.
    CritShkFuncList : [function]
        List of functions for each copay that map levels of expenditure xLvl to the critical
        medical shock at which effective consumption equals Cfloor when xLvl is spent.
    CubicBool: boolean
        An indicator for whether the solver should use cubic or linear interpolation.
                    
    Returns
    -------
    solution : InsuranceSelectionSolution
    '''
    t_start = clock()
    
    # Get sizes of arrays
    mLvl_aug_factor = 4
    MedShk_aug_factor = 4
    pLvlCount = pLvlGrid.size
    aLvlCount   = aXtraGrid.size
    HealthCount = len(LivPrb) # number of discrete health states
    
    # Make a JDfixer instance to use when necessary
    mGridDenseBase = makeGridDenser(aXtraGrid,mLvl_aug_factor)
    ShkGridDenseExample = makeGridDenser(MedShkDstn[0][1],MedShk_aug_factor)
    MyJDfixer = JDfixer(aLvlCount+1,MedShkDstn[0][1].size,mGridDenseBase.size,ShkGridDenseExample.size)
    
    # Define utility function derivatives and inverses
    u = lambda x : utility(x,CRRA)
    uP = lambda x : utilityP(x,CRRA)
    uPinv = lambda x : utilityP_inv(x,CRRA)
    uinvP = lambda x : utility_invP(x,CRRA)
    uinv = lambda x : utility_inv(x,CRRA)
    
    # For each future health state, find the minimum allowable end of period assets by permanent income    
    aLvlMinCond = np.zeros((pLvlCount,HealthCount))
    for h in range(HealthCount):
        # Unpack the inputs conditional on future health state
        PermShkValsNext  = IncomeDstn[h][1]
        TranShkValsNext  = IncomeDstn[h][2]
        PermShkMinNext   = np.min(PermShkValsNext)    
        TranShkMinNext   = np.min(TranShkValsNext)
        PermIncMinNext = PermShkMinNext*pLvlNextFunc(pLvlGrid)
        IncLvlMinNext  = PermIncMinNext*TranShkMinNext
        aLvlMinCond[:,h] = (solution_next.mLvlMin(PermIncMinNext) - IncLvlMinNext)/Rfree[h]
    aLvlMin = np.max(aLvlMinCond,axis=1) # Actual minimum acceptable assets is largest among health-conditional values
    
    # Make natural and artificial borrowing constraint and the constrained spending function
    if pLvlGrid[0] > 0.0:
        BoroCnstNat = LinearInterp(np.insert(pLvlGrid,0,0.0),np.insert(aLvlMin,0,0.0))
    else:
        BoroCnstNat = LinearInterp(pLvlGrid,aLvlMin)
    if BoroCnstArt is not None:
        BoroCnstArt = LinearInterp([0.0,1.0],[0.0,BoroCnstArt])
        mLvlMinNow = UpperEnvelope(BoroCnstNat,BoroCnstArt)
    else:
        mLvlMinNow = BoroCnstNat
    spendAllFunc = IdentityFunction(i_dim=0,n_dims=3)
    xFuncNowCnst = VariableLowerBoundFunc3D(spendAllFunc,mLvlMinNow)
    
    # For each future health state, calculate expected value and marginal value on grids of a and p
    EndOfPrdvCond = np.zeros((pLvlCount,aLvlCount+1,HealthCount))
    EndOfPrdvPcond = np.zeros((pLvlCount,aLvlCount+1,HealthCount))
    if CubicBool:
        EndOfPrdvPPcond = np.zeros((pLvlCount,aLvlCount,HealthCount))
    hLvlCond = np.zeros((pLvlCount,HealthCount))
    for h in range(HealthCount):
        # Unpack the inputs conditional on future health state
        ShkPrbsNext      = IncomeDstn[h][0]
        PermShkValsNext  = IncomeDstn[h][1]
        TranShkValsNext  = IncomeDstn[h][2]
        ShkCount         = PermShkValsNext.size
        vPfuncNext       = solution_next.vPfunc[h]      
        if CubicBool:
            vPPfuncNext  = solution_next.vPPfunc[h]            
        vFuncNext        = solution_next.vFunc[h]
        
        # Calculate human wealth conditional on achieving this future health state
        PermIncNext   = np.tile(pLvlNextFunc(pLvlGrid),(ShkCount,1))*np.tile(PermShkValsNext,(pLvlCount,1)).transpose()
        hLvlCond[:,h] = 1.0/Rfree[h]*np.sum((np.tile(TranShkValsNext,(pLvlCount,1)).transpose()*PermIncNext + solution_next.hLvl[h](PermIncNext))*np.tile(ShkPrbsNext,(pLvlCount,1)).transpose(),axis=0)
        
        # Make arrays of current end of period states
        aLvlGrid    = np.insert(aXtraGrid,0,0.0)
        pLvlNow     = np.tile(pLvlGrid,(aLvlCount+1,1)).transpose()
        aLvlNow     = np.tile(aLvlGrid,(pLvlCount,1))*pLvlNow #+ np.tile(aLvlMin,(aLvlCount,1)).transpose()
        pLvlNow_tiled = np.tile(pLvlNow,(ShkCount,1,1))
        aLvlNow_tiled = np.tile(aLvlNow,(ShkCount,1,1)) # shape = (ShkCount,pLvlCount,aLvlCount)
        if pLvlGrid[0] == 0.0:  # aLvl turns out badly if pLvl is 0 at bottom
            aLvlNow[0,:] = aLvlGrid*pLvlGrid[1]
            aLvlNow_tiled[:,0,:] = np.tile(aLvlGrid*pLvlGrid[1],(ShkCount,1))
            
        # Tile arrays of the income shocks and put them into useful shapes
        PermShkVals_tiled = np.transpose(np.tile(PermShkValsNext,(aLvlCount+1,pLvlCount,1)),(2,1,0))
        TranShkVals_tiled = np.transpose(np.tile(TranShkValsNext,(aLvlCount+1,pLvlCount,1)),(2,1,0))
        ShkPrbs_tiled     = np.transpose(np.tile(ShkPrbsNext,(aLvlCount+1,pLvlCount,1)),(2,1,0))
        
        # Make grids of future states conditional on achieving this future health state
        pLvlNext = pLvlNextFunc(pLvlNow_tiled)*PermShkVals_tiled
        mLvlNext = Rfree[h]*aLvlNow_tiled + pLvlNext*TranShkVals_tiled
        
        # Calculate future-health-conditional end of period value and marginal value
        tempv = vFuncNext(mLvlNext,pLvlNext)
        tempvP = vPfuncNext(mLvlNext,pLvlNext)
        EndOfPrdvPcond[:,:,h]  = Rfree[h]*np.sum(tempvP*ShkPrbs_tiled,axis=0)
        EndOfPrdvCond[:,:,h]   = np.sum(tempv*ShkPrbs_tiled,axis=0)
        if CubicBool:
            EndOfPrdvPPcond[:,:,h] = Rfree[h]*Rfree[h]*np.sum(vPPfuncNext(mLvlNext,pLvlNext)*ShkPrbs_tiled,axis=0)
        #print(np.argwhere(tempvP == 0.))
        
    # Calculate end of period value and marginal value conditional on each current health state
    EndOfPrdv = np.zeros((pLvlCount,aLvlCount+1,HealthCount))
    EndOfPrdvP = np.zeros((pLvlCount,aLvlCount+1,HealthCount))
    if CubicBool:
        EndOfPrdvPP = np.zeros((pLvlCount,aLvlCount,HealthCount))
    for h in range(HealthCount):
        # Set up a temporary health transition array
        HealthTran_temp = np.tile(np.reshape(MrkvArray[h,:],(1,1,HealthCount)),(pLvlCount,aLvlCount+1,1))
        DiscFacEff = DiscFac*LivPrb[h] # "effective" discount factor
        
        # Weight future health states according to the transition probabilities
        EndOfPrdv[:,:,h]  = DiscFacEff*np.sum(EndOfPrdvCond*HealthTran_temp,axis=2)
        EndOfPrdvP[:,:,h] = DiscFacEff*np.sum(EndOfPrdvPcond*HealthTran_temp,axis=2)
        if CubicBool:
            EndOfPrdvPP[:,:,h] = DiscFacEff*np.sum(EndOfPrdvPPcond*HealthTran_temp,axis=2)
            
    # Calculate human wealth conditional on each current health state 
    hLvlGrid = (np.dot(MrkvArray,hLvlCond.transpose())).transpose()
    
    # Loop through current health states to solve the period at each one
    solution_now = InsuranceSelectionSolution(mLvlMin=mLvlMinNow)
    JDfixCount = 0
    for h in range(HealthCount):
        MedShkPrbs       = MedShkDstn[h][0]
        MedShkVals       = MedShkDstn[h][1]
        MedShkMax        = np.exp(MedShkAvg[h] + MedShkStd[h]*6.0) # In search for CritShk, never go above this
        MedCount         = MedShkVals.size
        mCount           = EndOfPrdvP.shape[1]
        pCount           = EndOfPrdvP.shape[0]
        ZgridBase = (np.log(MedShkVals) - MedShkAvg[h])/MedShkStd[h] # Find baseline Z-grids for the medical shock distribution
        ShkGridDense = makeGridDenser(MedShkVals,MedShk_aug_factor)
        policyFuncsThisHealthCopay = []
        vFuncsThisHealthCopay = []
#        CritShkFuncsThisHealthCopay = []
        
        # Make an alternate shock and prob array for actuarial value calculation
        MedShkArrayAlt = np.tile(np.reshape(MedShkVals,(1,1,MedCount)),(aLvlCount,pLvlCount,1))
        ShkPrbsArrayAlt = np.tile(np.reshape(MedShkPrbs,(1,1,MedCount)),(aLvlCount,pLvlCount,1))
        
        # Make the end of period value function for this health
        EndOfPrdvNvrsFunc_by_pLvl = []
        EndOfPrdvNvrs = uinv(EndOfPrdv[:,:,h])
        for j in range(pLvlCount):
            a_temp = aLvlNow[j,:]
            EndOfPrdvNvrs_temp = EndOfPrdvNvrs[j,:]
            EndOfPrdvNvrsFunc_by_pLvl.append(LinearInterp(a_temp,EndOfPrdvNvrs_temp))
        EndOfPrdvNvrsFuncBase = LinearInterpOnInterp1D(EndOfPrdvNvrsFunc_by_pLvl,pLvlGrid)
        EndOfPrdvNvrsFunc = EndOfPrdvNvrsFuncBase
        EndOfPrdvFunc = ValueFunc2D(EndOfPrdvNvrsFunc,CRRA)
        
        # Calculate end-of-period value from ending with a=0, and calculate vFloor by pLvl
        EndOfPrdv_if_aLvl_zero = EndOfPrdvFunc(np.zeros_like(pLvlGrid),pLvlGrid)
        vFloorBypLvl = u(Cfloor) + EndOfPrdv_if_aLvl_zero
        
        # For each coinsurance rate, make policy and value functions (for this health state)
        EndOfPrdvPnvrs_tiled = np.tile(np.reshape(uPinv(EndOfPrdvP[:,:,h]),(1,pCount,mCount)),(MedCount,1,1))
        MedShkVals_tiled  = np.tile(np.reshape(MedShkVals,(MedCount,1,1)),(1,pCount,mCount))
        for k in range(len(CopayList)):
            MedPriceEff = CopayList[k]
            Dfunc = DfuncList[k]
            Gfunc = GfuncList[k]
            cEffFunc = cEffFuncList[k]
            
            # Calculate endogenous gridpoints and controls
            cEffNow = Gfunc(EndOfPrdvPnvrs_tiled,MedShkVals_tiled)
            ShkZero = MedShkVals_tiled == 0.
            cEffNow[ShkZero] = EndOfPrdvPnvrs_tiled[ShkZero]
            xLvlNow = Dfunc(cEffNow,MedShkVals_tiled)
            aLvlNow_tiled = np.tile(np.reshape(aLvlNow,(1,pCount,mCount)),(MedCount,1,1))
            mLvlNow = xLvlNow + aLvlNow_tiled
            #print(np.sum(np.isnan(EndOfPrdvPnvrs_tiled)),np.sum(np.isinf(EndOfPrdvPnvrs_tiled)))
            
            # Calculate marginal propensity to spend
            if CubicBool:
                print("CubicBool=True doesn't work at this time")
                
            # Loop over each permanent income level and medical shock and make an xFunc
            NonMonotonic = np.any((xLvlNow[:,:,1:]-xLvlNow[:,:,:-1]) < 0.,axis=(0,2)) # whether each pLvl has non-monotonic pattern in mLvl gridpoints
            HasNaNs = np.any(np.isnan(xLvlNow),axis=(0,2)) # whether each pLvl contains any NaNs due to future vP=0.0
            NeedsJDfix = np.logical_or(NonMonotonic,HasNaNs)
            JDfixCount += np.sum(NeedsJDfix)
            xFunc_by_pLvl = [] # Initialize the empty list of lists of 1D xFuncs
            for i in range(pCount):
                if NeedsJDfix[i]:
                    #t0 = clock()
                    aLvl_temp = aLvlNow_tiled[:,i,:]
                    pLvl_temp = pLvlGrid[i]*np.ones_like(aLvl_temp)
                    vNvrs_data = (uinv(u(cEffNow[:,i,:]) + EndOfPrdvFunc(aLvl_temp,pLvl_temp))).transpose()
                    mLvl_data = mLvlNow[:,i,:].transpose()
                    xLvl_data = xLvlNow[:,i,:].transpose()
                    MedShk_data = MedShkVals_tiled[:,i,:].transpose()
                    pLvl_i = pLvlGrid[i]
                    if pLvl_i == 0.0:
                        pLvl_i = pLvlGrid[i+1]
                    mGridDense = mGridDenseBase*pLvl_i
                    xFunc_by_pLvl.append(MyJDfixer(mLvl_data,MedShk_data,vNvrs_data,xLvl_data,mGridDense,ShkGridDense))
                    mLvlNow[:,i,0] = 0.0 # This fixes the "seam" problem so there are no NaNs
                    #t1 = clock()
                    #print('JD fix took ' + str(t1-t0) + ' seconds.')
                else:
                    temp_list = []
                    for j in range(MedCount):
                        m_temp = mLvlNow[j,i,:] - mLvlNow[j,i,0]
                        x_temp = xLvlNow[j,i,:]
                        temp_list.append(LinearInterp(m_temp,x_temp))
                    xFunc_by_pLvl.append(LinearInterpOnInterp1D(temp_list,MedShkVals))
            
            # Combine the many expenditure functions into a single one and adjust for the natural borrowing constraint
            ConstraintSeam = BilinearInterp((mLvlNow[:,:,0]).transpose(),pLvlGrid,MedShkVals)
            xFuncNowUncBase = TwistFuncB(LinearInterpOnInterp2D(xFunc_by_pLvl,pLvlGrid))
            xFuncNowUnc = VariableLowerBoundFunc3Dalt(xFuncNowUncBase,ConstraintSeam)
            xFuncNow = CompositeFunc3D(xFuncNowCnst,xFuncNowUnc,ConstraintSeam)
            
            # Make a policy function for this coinsurance rate and health state
            policyFuncsThisHealthCopay.append(MedShockPolicyFuncPrecalc(xFuncNow,bFromxFuncList[k],MedPriceEff))
            
            # Calculate pseudo inverse value on a grid of states for this coinsurance rate
            pLvlArray = np.tile(np.reshape(pLvlGrid,(1,pCount,1)),(MedCount,1,mCount-1))
            mMinArray = np.tile(np.reshape(mLvlMinNow(pLvlGrid),(1,pCount,1)),(MedCount,1,mCount-1))
            mLvlArray = mMinArray + np.tile(np.reshape(aXtraGrid,(1,1,mCount-1)),(MedCount,pCount,1))*pLvlArray
            if pLvlGrid[0] == 0.0:  # mLvl turns out badly if pLvl is 0 at bottom
                mLvlArray[:,0,:] = np.tile(aXtraGrid,(MedCount,1))
            MedShkArray = np.tile(np.reshape(MedShkVals,(MedCount,1,1)),(1,pCount,mCount-1))
            xLvlArray = xFuncNow(mLvlArray,pLvlArray,MedShkArray)
            cEffLvlArray = cEffFunc(xLvlArray,MedShkArray)
            aLvlArray = np.abs(mLvlArray - xLvlArray) # OCCASIONAL VIOLATIONS BY 1E-18 !!!
            vNow = u(cEffLvlArray) + EndOfPrdvFunc(aLvlArray,pLvlArray)
            vNvrsNow  = np.concatenate((np.zeros((MedCount,pCount,1)),uinv(vNow)),axis=2)
                
            # Loop over each permanent income level and mLvl and make a vNvrsFunc over MedShk for each
            vNvrsFunc_by_pLvl = [] # Initialize the empty list of lists of vNvrsFuncs
            for i in range(pCount):
                temp_list = [ConstantFunction(0.0)] # Initialize for mLvl=0 --> vNvrs=0
                for j in range(1,mCount):
                    vNvrsLog_temp = np.log(vNvrsNow[:,i,j])
                    temp_list.append(LogFunc1D(LinearInterp(MedShkVals,vNvrsLog_temp)))
                m_temp = np.insert(mLvlArray[0,i,:],0,0.0) # Combine across mLvl within this pLvl
                vNvrsFunc_by_pLvl.append(LinearInterpOnInterp1D(temp_list,m_temp))
            vNvrsFuncBase = LinearInterpOnInterp2D(vNvrsFunc_by_pLvl,pLvlGrid) # Combine across all pLvls
            vNvrsFunc = TwistFuncA(vNvrsFuncBase) # Change input order from (MedShk,mLvl,pLvl) to (mLvl,pLvl,MedShk)
            
            # Add the value function to the list for this health
            vFuncsThisHealthCopay.append(ValueFunc3D(vNvrsFunc,CRRA)) # Recurve value function
            
#            # Find the critical med shock where Cfloor binds for each (mLvl,pLvl) value
#            xFunc_temp = xFuncNow # Assume zero deductible for now, will fix later
#            CritShkFunc = CritShkFuncList[k]
#            mMinArray_temp = np.tile(np.reshape(mLvlMinNow(pLvlGrid),(1,pLvlCount)),(aLvlCount,1))
#            pLvlArray_temp = np.tile(np.reshape(pLvlGrid,(1,pLvlCount)),(aLvlCount,1))
#            mLvlArray_temp = mMinArray_temp + np.tile(np.reshape(aXtraGrid,(aLvlCount,1)),(1,pLvlCount))*pLvlArray_temp + Cfloor
#            CritShkArray = 1e-8*np.ones_like(mLvlArray_temp) # Current guess of critical shock for each (mLvl,pLvl)
#            DiffArray = np.ones_like(mLvlArray_temp) # Relative change in crit shock guess this iteration
#            Unresolved = np.ones_like(mLvlArray_temp,dtype=bool) # Indicator for which points are still unresolved
#            UnresolvedCount = Unresolved.size # Number of points whose CritShk has not been found
#            DiffTol = 1e-5 # Convergence tolerance for the search
#            LoopCount = 0
#            LoopMax = 30
#            while (UnresolvedCount > 0) and (LoopCount < LoopMax): # Loop until all points have converged on CritShk
#                CritShkPrev = CritShkArray[Unresolved]
#                mLvl_temp = mLvlArray_temp[Unresolved]                
#                if LoopCount > 30: # Use Newton's method after a few iterations
#                    xLvl_temp = np.minimum(xFunc_temp(mLvl_temp,pLvlArray_temp[Unresolved],CritShkPrev),mLvl_temp)
#                    CritShk_temp = np.minimum(CritShkFunc(xLvl_temp),MedShkMax)
#                    Target_diff = CritShk_temp - CritShkPrev # This is the expression we want to be zero
#                    Target_slope = xFunc_temp.derivativeZ(mLvl_temp,pLvlArray_temp[Unresolved],CritShkPrev)*CritShkFunc.derivative(xLvl_temp) - 1.
#                    CritShkNew = CritShkPrev - Target_diff/Target_slope
#                    DiffNew = np.abs(CritShkNew/CritShkPrev - 1.)
#                else: # Use fixed point iteration for first few iterations
#                    xLvl_temp = np.minimum(xFunc_temp(mLvl_temp,pLvlArray_temp[Unresolved],CritShkPrev),mLvl_temp)
#                    CritShkNew = np.minimum(CritShkFunc(xLvl_temp),MedShkMax)
#                    DiffNew = np.abs(CritShkNew/CritShkPrev - 1.)                
#                DiffArray[Unresolved] = DiffNew
#                CritShkArray[Unresolved] = CritShkNew
#                Unresolved[Unresolved] = DiffNew > DiffTol
#                UnresolvedCount = np.sum(Unresolved)
#                LoopCount += 1
#                
#            # Construct a function that yields the critical medical shock where the Cfloor begins to bind (for given mLvl,pLvl)
#            CritShkFunc_by_pLvl = []
#            for i in range(pLvlCount):
#                m_temp = np.insert(mLvlArray_temp[:,i],0,Cfloor)
#                Shk_temp = np.insert(CritShkArray[:,i],0,0.0)
#                CritShkFunc_by_pLvl.append(LinearInterp(m_temp,Shk_temp))
#            CritShkFuncsThisHealthCopay.append(LinearInterpOnInterp1D(CritShkFunc_by_pLvl,pLvlGrid))            
            
        # Set up state grids to prepare for the medical shock integration step
        tempArray    = np.tile(np.reshape(aXtraGrid,(aLvlCount,1,1)),(1,pLvlCount,MedCount))
        mMinArray    = np.tile(np.reshape(mLvlMinNow(pLvlGrid),(1,pLvlCount,1)),(aLvlCount,1,MedCount))
        pLvlArray    = np.tile(np.reshape(pLvlGrid,(1,pLvlCount,1)),(aLvlCount,1,MedCount))
        mLvlArray    = mMinArray + tempArray*pLvlArray + Cfloor
        if pLvlGrid[0] == 0.0: # Fix the problem of all mLvls = 0 when pLvl = 0
            mLvlArray[:,0,:] = mLvlArray[:,1,:]
        
        # For each insurance contract available in this health state, "integrate" across medical need
        # shocks to get policy and (marginal) value functions for each contract
        policyFuncsThisHealth = []
        vFuncsThisHealth = []
        vPfuncsThisHealth = []
        AVfuncsThisHealth = []
        for z in range(len(ContractList[h])):
            # Set and unpack the contract of interest
            Contract = ContractList[h][z]
            Copay = Contract.Copay
            FullPrice_idx = np.argwhere(np.array(CopayList)==MedPrice)[0][0]
            Copay_idx = np.argwhere(np.array(CopayList)==Copay*MedPrice)[0][0]
            
            # Get the value and policy functions for this contract
            vFuncFullPrice = vFuncsThisHealthCopay[FullPrice_idx]
            vFuncCopay = vFuncsThisHealthCopay[Copay_idx]
            policyFuncFullPrice = policyFuncsThisHealthCopay[FullPrice_idx]
            policyFuncCopay = policyFuncsThisHealthCopay[Copay_idx]
            DpFuncFullPrice = DpFuncList[FullPrice_idx]
            DpFuncCopay = DpFuncList[Copay_idx]
            CritShkFuncFullPrice = CritShkFuncList[FullPrice_idx]
            CritShkFuncCopay = CritShkFuncList[Copay_idx]
            
            # Make the policy function for this contract
            policyFuncsThisHealth.append(InsSelPolicyFunc(vFuncFullPrice,vFuncCopay,policyFuncFullPrice,policyFuncCopay,DpFuncFullPrice,DpFuncCopay,Contract,CRRAmed))
            
            # Find the critical med shock where Cfloor binds for each (mLvl,pLvl) value
            xFunc_temp = policyFuncsThisHealth[-1].xFunc
            mLvlArray_temp = mLvlArray[:,:,0]
            pLvlArray_temp = pLvlArray[:,:,0]
            CritShkArray = 1e-8*np.ones_like(mLvlArray_temp) # Current guess of critical shock for each (mLvl,pLvl)
            DiffArray = np.ones_like(mLvlArray_temp) # Relative change in crit shock guess this iteration
            Unresolved = np.ones_like(mLvlArray_temp,dtype=bool) # Indicator for which points are still unresolved
            UnresolvedCount = Unresolved.size # Number of points whose CritShk has not been found
            DiffTol = 1e-5 # Convergence tolerance for the search
            LoopCount = 0
            LoopMax = 30
            while (UnresolvedCount > 0) and (LoopCount < LoopMax): # Loop until all points have converged on CritShk
                CritShkPrev = CritShkArray[Unresolved]
                mLvl_temp = mLvlArray_temp[Unresolved]                
#                if LoopCount > 30: # Use Newton's method after a few iterations
#                    xLvl_temp = np.minimum(xFunc_temp(mLvl_temp,pLvlArray_temp[Unresolved],CritShkPrev),mLvl_temp)
#                    CritShk_temp = np.minimum(CritShkFunc(xLvl_temp),MedShkMax)
#                    Target_diff = CritShk_temp - CritShkPrev # This is the expression we want to be zero
#                    Target_slope = xFunc_temp.derivativeZ(mLvl_temp,pLvlArray_temp[Unresolved],CritShkPrev)*CritShkFunc.derivative(xLvl_temp) - 1.
#                    CritShkNew = CritShkPrev - Target_diff/Target_slope
#                    DiffNew = np.abs(CritShkNew/CritShkPrev - 1.)
#                else: # Use fixed point iteration for first few iterations
                xLvl_temp, Copay_bool_temp = xFunc_temp(mLvl_temp,pLvlArray_temp[Unresolved],CritShkPrev,return_copay_bool=True)
                xLvl_temp = np.minimum(xLvl_temp,mLvl_temp)
                CritShkNew = np.zeros_like(xLvl_temp)
                FullPrice_bool_temp = np.logical_not(Copay_bool_temp)
                CritShkNew[Copay_bool_temp] = np.minimum(CritShkFuncCopay(xLvl_temp[Copay_bool_temp]),MedShkMax)
                CritShkNew[FullPrice_bool_temp] = np.minimum(CritShkFuncFullPrice(xLvl_temp[FullPrice_bool_temp]),MedShkMax)
                DiffNew = np.abs(CritShkNew/CritShkPrev - 1.)
                
                DiffArray[Unresolved] = DiffNew
                CritShkArray[Unresolved] = CritShkNew
                Unresolved[Unresolved] = DiffNew > DiffTol
                UnresolvedCount = np.sum(Unresolved)
                LoopCount += 1
#            if LoopCount == LoopMax:
#                print(str(UnresolvedCount) + ' points still unresolved for h=' + str(h) + ', z=' + str(z) + '!')

#            # Find the critical med shock where Cfloor binds for each (mLvl,pLvl) value
#            mLvlArray_temp = mLvlArray[:,:,0]
#            pLvlArray_temp = pLvlArray[:,:,0]
#            CritShkFunc = CritShkFuncsThisHealthCopay[Copay_idx]
#            CritShkArray = CritShkFunc(mLvlArray_temp,pLvlArray_temp)
                
            # Make arrays of medical shocks and probabilities, along with Cfloor probs and vFloor values
            AlwaysCfloor = np.logical_not(CritShkArray > 0.)
#            NeverCfloor = CritShkArray == MedShkMax # If Cfloor is hit less than a billionth of the time, call it "never"
            ZcritArray = (np.log(CritShkArray) - MedShkAvg[h])/MedShkStd[h]
            CfloorPrbArray = norm.sf(ZcritArray)*(1.-MedShkPrbs[0])
            CfloorPrbArray[AlwaysCfloor] = 1.0 # These were shifted down by line above, but that's wrong
            ZadjArray = np.minimum(ZcritArray - ZgridBase[-1],0.) # Should always be non-positive
            ZshkArray = np.tile(np.reshape(ZgridBase,(1,1,MedCount)),(aLvlCount,pLvlCount,1)) + np.tile(np.reshape(ZadjArray,(aLvlCount,pLvlCount,1)),(1,1,MedCount))
            MedShkArray = np.exp(ZshkArray*MedShkStd[h] + MedShkAvg[h])

#            LogMedShkLower = MedShkAvg[h] - 3.0*MedShkStd[h]
#            LogCritShkArray = np.log(CritShkArray)
#            FracArray = np.tile(np.reshape(np.linspace(0.0,1.0,MedCount),(1,1,MedCount)),(aLvlCount,pLvlCount,1))
#            LogMedShkArray = LogMedShkLower + FracArray*(np.tile(np.reshape(LogCritShkArray,(aLvlCount,pLvlCount,1)),(1,1,MedCount)) - LogMedShkLower)
#            MedShkArray = np.exp(LogMedShkArray)
#            ZshkArray = (LogMedShkArray - MedShkAvg[h])/MedShkStd[h]
            
            TempPrbArray = norm.pdf(ZshkArray)
            ReweightArray = (1.-MedShkPrbs[0]-CfloorPrbArray)/np.sum(TempPrbArray,axis=2)
            ShkPrbsArray = TempPrbArray*np.tile(np.reshape(ReweightArray,(aLvlCount,pLvlCount,1)),(1,1,MedCount))
            ShkPrbsArray[:,:,0] = MedShkPrbs[0]
            AlwaysCfloor_tiled = np.tile(np.reshape(AlwaysCfloor,(aLvlCount,pLvlCount,1)),(1,1,MedCount))
            ShkPrbsArray[AlwaysCfloor_tiled] = 0.0
                           
            # Get value and marginal value at an array of states
            vArrayBig, vParrayBig, vPParrayBig = policyFuncsThisHealth[-1].evalvAndvPandvPP(mLvlArray,pLvlArray,MedShkArray)
            ConArrayBig, MedArrayBig = policyFuncsThisHealth[-1](mLvlArray,pLvlArray,MedShkArrayAlt)
            
            # Fix tiny non-monotonicities in value near the Cfloor seam
            vFloor_tiled = np.tile(np.reshape(vFloorBypLvl,(1,pLvlCount,1)),(aLvlCount,1,MedCount))
            vArrayBig = np.maximum(vArrayBig,vFloor_tiled)
                        
            # Integrate (marginal) value across medical shocks
            vArray   = np.sum(vArrayBig*ShkPrbsArray,axis=2) + CfloorPrbArray*np.tile(np.reshape(vFloorBypLvl,(1,pLvlCount)),(aLvlCount,1))
            vParray  = np.sum(vParrayBig*ShkPrbsArray,axis=2)
                
#            # Make a second array of shocks and probabilities *beyond* the critical shock (only relevant for AV)
#            ZadjAltArray = np.maximum(ZcritArray - ZgridBase[1],0.) # Should always be non-negative
#            ZshkAltArray = np.minimum(np.tile(np.reshape(ZgridBase[1:],(1,1,MedCount-1)),(aLvlCount,pLvlCount,1)) + np.tile(np.reshape(ZadjAltArray,(aLvlCount,pLvlCount,1)),(1,1,MedCount-1)),6.)
#            MedShkAltArray = np.exp(ZshkAltArray*MedShkStd[h] + MedShkAvg[h])
#            TempPrbAltArray = norm.pdf(ZshkAltArray)
#            ReweightAltArray = np.sum(TempPrbAltArray,axis=2)
#            ShkPrbsAltArray = TempPrbAltArray*np.tile(np.reshape(ReweightAltArray,(aLvlCount,pLvlCount,1)),(1,1,MedCount-1))
#            
#            # Find medical care at each "alternative shock" beyond the critical shock
#            xLvlAlt = DfuncList[Copay_idx](Cfloor*np.ones_like(MedShkAltArray),MedShkAltArray)
#            b_temp  = bFromxFuncList[Copay_idx](xLvlAlt,MedShkAltArray) # transformed consumption ratio
#            q_temp  = np.exp(-b_temp)
#            MedArrayAlt = xLvlAlt*q_temp/(1.+q_temp)
            
            # Calculate actuarial value at each (mLvl,pLvl), combining shocks above and below the critical value
            AVarrayBig = MedArrayBig*MedPrice - Contract.OOPfunc(MedArrayBig) # realized "actuarial value" below critical shock
#            AVarrayAlt = MedArrayAlt*MedPrice - Contract.OOPfunc(MedArrayAlt) # realized "actuarial value" above critical shock
            AVarray  = np.sum(AVarrayBig*ShkPrbsArrayAlt,axis=2)# + CfloorPrbArray*np.sum(AVarrayAlt*ShkPrbsAltArray,axis=2)
            
            # Construct pseudo-inverse arrays of vNvrs and vPnvrs, adding some data at the bottom
            vNvrsFloorBypLvl = uinv(vFloorBypLvl)
            mLvlBound = mLvlArray_temp[0,:] # Lowest mLvl for each pLvl; will be used to make "seam"
            mGrid_small_A = np.concatenate((np.zeros((1,pLvlCount)),Cfloor*np.ones((1,pLvlCount)),mLvlArray_temp),axis=0) # for value
            mGrid_small_B = mLvlArray_temp - np.tile(np.reshape(mLvlBound,(1,pLvlCount)),(aLvlCount,1)) # for marginal value
            mGrid_small_C = np.concatenate((np.zeros((1,pLvlCount)),mLvlArray_temp),axis=0) # for AV
            vPnvrsArray   = uPinv(vParray)
            vNvrsArray    = np.concatenate((np.tile(np.reshape(vNvrsFloorBypLvl,(1,pLvlCount)),(2,1)),uinv(vArray)),axis=0)
            
            # Construct the pseudo-inverse value and marginal value functions over mLvl,pLvl
            vNvrsFunc_by_pLvl = []
            vPnvrsFuncUpper_by_pLvl = []
            vPfuncLower_by_pLvl = []
            AVfunc_by_pLvl = []
            for j in range(pLvlCount): # Make a pseudo inverse marginal value function for each pLvl
                vPfuncLower_by_pLvl.append(LinearInterp(np.array([0.,Cfloor,mLvlBound[j]]),np.array([0.,0.,vParray[0,j]])))
                m_temp_A = mGrid_small_A[:,j]
                m_temp_B = mGrid_small_B[:,j]
                m_temp_C = mGrid_small_C[:,j]
                vNvrs_temp  = vNvrsArray[:,j]
                vPnvrs_temp = vPnvrsArray[:,j]
                AV_temp = np.insert(AVarray[:,j],0,0.0)
                vNvrsFunc_by_pLvl.append(LinearInterp(m_temp_A,vNvrs_temp))
                vPnvrsFuncUpper_by_pLvl.append(LinearInterp(m_temp_B,vPnvrs_temp))
                AVfunc_by_pLvl.append(LinearInterp(m_temp_C,AV_temp))
            vNvrsFuncBase  = LinearInterpOnInterp1D(vNvrsFunc_by_pLvl,pLvlGrid)
            vNvrsFunc = VariableLowerBoundFunc2D(vNvrsFuncBase,mLvlMinNow) # adjust for the lower bound of mLvl
            AVfuncBase = LinearInterpOnInterp1D(AVfunc_by_pLvl,pLvlGrid)
            AVfunc = VariableLowerBoundFunc2D(AVfuncBase,mLvlMinNow) # adjust for the lower bound of mLvl
                
            # Build the marginal value function by putting together the lower and upper portions
            vPfuncSeam  = LinearInterp(pLvlGrid,mLvlBound,lower_extrap=True)
            vPnvrsFuncUpperBase = LinearInterpOnInterp1D(vPnvrsFuncUpper_by_pLvl,pLvlGrid)
            vPnvrsFuncUpper = VariableLowerBoundFunc2D(vPnvrsFuncUpperBase,vPfuncSeam) # adjust for the lower bound of mLvl
            vPfuncUpper = MargValueFunc2D(vPnvrsFuncUpper,CRRA)
            vPfuncLower = LinearInterpOnInterp1D(vPfuncLower_by_pLvl,pLvlGrid)
            
            # Store the policy and (marginal) value function 
            vFuncsThisHealth.append(ValueFunc2D(vNvrsFunc,CRRA))
            vPfuncsThisHealth.append(CompositeFunc2D(vPfuncLower,vPfuncUpper,vPfuncSeam))
            AVfuncsThisHealth.append(AVfunc)
        
        # If there is only one contract, then value and marginal value functions are trivial.
        if len(ContractList[h]) == 1:
            vFunc = vFuncsThisHealth[0] # only element
            vPfunc = vPfuncsThisHealth[0] # only element
        else: # If there is more than one contract available, take expectation over choice shock.
            mLvlGrid = mGridDenseBase
            mLvlCount = mLvlGrid.size
        
            # Make grids to prepare for the choice shock step
            tempArray    = np.tile(np.reshape(mLvlGrid,(mLvlCount,1)),(1,pLvlCount))
            mMinArray    = np.tile(np.reshape(mLvlMinNow(pLvlGrid),(1,pLvlCount)),(mLvlCount,1))
            pLvlArray    = np.tile(np.reshape(pLvlGrid,(1,pLvlCount)),(mLvlCount,1))
            mLvlArray    = mMinArray + tempArray*pLvlArray + Cfloor
            if pLvlGrid[0] == 0.0: # Fix the problem of all mLvls = 0 when pLvl = 0
                mLvlArray[:,0] = mLvlArray[:,1]
            
            # Get value and marginal value at each point in mLvl x pLvl at each contract, taking account of premiums paid
            vArrayBig   = np.zeros((mLvlCount,pLvlCount,len(ContractList[h]))) # value at each gridpoint for each contract
            vParrayBig  = np.zeros((mLvlCount,pLvlCount,len(ContractList[h]))) # marg value at each gridpoint for each contract
            vPParrayBig = np.zeros((mLvlCount,pLvlCount,len(ContractList[h]))) # marg marg value at each gridpoint for each contract
            AdjusterArray = np.zeros((mLvlCount,pLvlCount,len(ContractList[h]))) # (1 - dPremium/dmLvl), usually equals 1
            UnaffordableArray = np.zeros((mLvlCount,pLvlCount,len(ContractList[h])),dtype=bool)
            for z in range(len(ContractList[h])):
                Contract = ContractList[h][z]
                PremiumArray = Contract.Premium(mLvlArray,pLvlArray)
                AdjusterArray[:,:,z] = 1.0 - Contract.Premium.derivativeX(mLvlArray,pLvlArray)
                mLvlArray_temp = mLvlArray-PremiumArray
                UnaffordableArray[:,:,z] = mLvlArray_temp <= 0
                vArrayBig[:,:,z]   = vFuncsThisHealth[z](mLvlArray_temp,pLvlArray)
                vParrayBig[:,:,z]  = vPfuncsThisHealth[z](mLvlArray_temp,pLvlArray)
                if CubicBool:
                    vPParrayBig[:,:,z] = vPfuncsThisHealth[z].derivativeX(mLvlArray_temp,pLvlArray)
                      
            # Transform value (etc) into the pseudo-inverse forms needed
            vNvrsArrayBig  = uinv(vArrayBig)
            vNvrsParrayBig = AdjusterArray*vParrayBig*uinvP(vArrayBig)
            vPnvrsArrayBig = uPinv(AdjusterArray*vParrayBig)
                
            # Fix the unaffordable points so they don't generate NaNs near bottom
            vNvrsArrayBig[UnaffordableArray] = -np.inf
            vNvrsParrayBig[UnaffordableArray] = 0.0
            vPnvrsArrayBig[UnaffordableArray] = 0.0
            
            # Weight each gridpoint by its contract probabilities
            if ChoiceShkMag < 0.0: # Never use choice shocks during solution
                v_best = np.max(vNvrsArrayBig,axis=2)
                v_best_tiled = np.tile(np.reshape(v_best,(mLvlCount,pLvlCount,1)),(1,1,len(ContractList[h])))
                vNvrsArrayBig_adjexp = np.exp((vNvrsArrayBig - v_best_tiled)/ChoiceShkMag)
                vNvrsArrayBig_adjexpsum = np.sum(vNvrsArrayBig_adjexp,axis=2)
                vNvrsArray = ChoiceShkMag*np.log(vNvrsArrayBig_adjexpsum) + v_best
                ContractPrbs = vNvrsArrayBig_adjexp/np.tile(np.reshape(vNvrsArrayBig_adjexpsum,(mLvlCount,pLvlCount,1)),(1,1,len(ContractList[h])))
                vNvrsParray = np.sum(ContractPrbs*vNvrsParrayBig,axis=2)
                vPnvrsArray = uPinv(vNvrsParray)*vNvrsArray
            else:
                z_choice = np.argmax(vNvrsArrayBig,axis=2)
                vNvrsArray = np.zeros((mLvlCount,pLvlCount))
                vPnvrsArray = np.zeros((mLvlCount,pLvlCount))
                for z in range(len(ContractList[h])): ### THIS CAN BE ACCELERATED WITHOUT THE LOOP #####
                    these = z_choice == z
                    vNvrsArray[these] = vNvrsArrayBig[:,:,z][these]
                    vPnvrsArray[these] = vPnvrsArrayBig[:,:,z][these]
                        
            # Make value and marginal value functions for the very beginning of the period, before choice shocks are drawn
            vNvrsArray_plus = np.concatenate((np.tile(np.reshape(vNvrsFloorBypLvl,(1,pLvlCount)),(2,1)),vNvrsArray),axis=0)
            vNvrsFuncs_by_pLvl = []
            vPfuncs_Lower_by_pLvl = []
            vPnvrsFuncs_Upper_by_pLvl = []
            m_extra = np.array([0.,Cfloor]) # Add this onto m_temp when making vNvrsFunc
            for j in range(pLvlCount): # Make 1D functions by pLvl
                m_temp_A = np.concatenate((m_extra,mLvlArray[:,j])) # For vNvrs
                m_temp_B = mLvlArray[:,j] - mLvlArray[0,j] # For vPnvrsUpper
                vNvrs_temp = vNvrsArray_plus[:,j]
                vNvrsFuncs_by_pLvl.append(LinearInterp(m_temp_A,vNvrs_temp))
                vPnvrs_temp = vPnvrsArray[:,j]
                vPnvrsFuncs_Upper_by_pLvl.append(LinearInterp(m_temp_B,vPnvrs_temp))
                vPfuncs_Lower_by_pLvl.append(LinearInterp(np.array([0.,Cfloor,mLvlArray[0,j]]),np.array([0.,0.,uP(vPnvrs_temp[0])])))
            
            # Combine across pLvls and re-curve the value function
            vNvrsFuncBase  = LinearInterpOnInterp1D(vNvrsFuncs_by_pLvl,pLvlGrid)
            vNvrsFunc      = VariableLowerBoundFunc2D(vNvrsFuncBase,mLvlMinNow)
            vFunc          = ValueFunc2D(vNvrsFunc,CRRA)
            
            # Construct the marginal value function by combining the upper and lower portions
            vPfuncSeam = LinearInterp(pLvlGrid,mLvlArray[0,:],lower_extrap=True)
            vPnvrsFuncUpperBase = LinearInterpOnInterp1D(vPnvrsFuncs_Upper_by_pLvl,pLvlGrid)
            vPnvrsFuncUpper     = VariableLowerBoundFunc2D(vPnvrsFuncUpperBase,vPfuncSeam)
            vPfuncUpper         = MargValueFunc2D(vPnvrsFuncUpper,CRRA)
            vPfuncLower         = LinearInterpOnInterp1D(vPfuncs_Lower_by_pLvl,pLvlGrid)
            vPfunc              = CompositeFunc2D(vPfuncLower,vPfuncUpper,vPfuncSeam)
            
        # Make the human wealth function for this health state and add solution to output
        hLvl_h = LinearInterp(np.insert(pLvlGrid,0,0.0),np.insert(hLvlGrid[:,h],0,0.0))
        solution_now.appendSolution(vFunc=vFunc,vPfunc=vPfunc,hLvl=hLvl_h,AVfunc=AVfuncsThisHealth,
                                        policyFunc=policyFuncsThisHealth,vFuncByContract=vFuncsThisHealth)
    
    # Return the solution for this period
    t_end = clock()
    #print('Solving a period of the problem took ' + str(t_end-t_start) + ' seconds, fix count = ' + str(JDfixCount))
    return solution_now
    
####################################################################################################
    
class InsSelConsumerType(MedShockConsumerType,MarkovConsumerType):
    '''
    Class for representing consumers in the insurance selection model.  Each period, they receive
    shocks to their discrete health state and permanent and transitory income; after choosing an
    insurance contract, they learn their medical need shock and choose levels of consumption and
    medical care.
    '''
    _time_vary = ['DiscFac','LivPrb','MedPrice','ContractList','MrkvArray','ChoiceShkMag','MedShkAvg','MedShkStd']
    _time_inv = ['CRRA','CRRAmed','Cfloor','Rfree','BoroCnstArt','CubicBool']
    
    def __init__(self,cycles=1,time_flow=True,**kwds):
        '''
        Instantiate a new InsSelConsumerType with given data, and construct objects
        to be used during solution (income distribution, assets grid, etc).
        See ConsumerParameters.init_med_shock for a dictionary of the keywords
        that should be passed to the constructor.
        
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
        # Initialize a basic AgentType
        AgentType.__init__(self,solution_terminal=deepcopy(self.solution_terminal_),
                           cycles=cycles,time_flow=time_flow,pseudo_terminal=False,**kwds)
        self.solveOnePeriod = solveInsuranceSelection # Choose correct solver
        self.poststate_vars = ['aLvlNow']
        self.time_vary = copy(InsSelConsumerType._time_vary)
        self.time_inv = copy(InsSelConsumerType._time_inv)
        
    def updateMedShockProcess(self):
        '''
        Constructs discrete distributions of medical preference shocks for each
        period in the cycle for each health state.  Distributions are saved as attribute MedShkDstn,
        which is added to time_vary.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        None
        '''
        MedShkDstn = [] # empty list for medical shock distribution each period
        for t in range(self.T_cycle):
            temp_list = []
            for h in range(self.MrkvArray[0].shape[0]):
                MedShkAvgNow  = self.MedShkAvg[t][h] # get shock distribution parameters
                MedShkStdNow  = self.MedShkStd[t][h]
                MedShkDstnNow = approxLognormal(mu=MedShkAvgNow,sigma=MedShkStdNow,N=self.MedShkCount, tail_N=self.MedShkCountTail, 
                                tail_bound=self.MedShkTailBound)
                ZeroMedShkPrbNow = self.ZeroMedShkPrb[t,h]
                MedShkDstnNow = addDiscreteOutcome(MedShkDstnNow,x=0.0,p=ZeroMedShkPrbNow,sort=True) # add point at zero with probability given by ZeroMedShkPrbNow
                temp_list.append(MedShkDstnNow)
            MedShkDstn.append(deepcopy(temp_list))
        self.MedShkDstn = MedShkDstn
        self.addToTimeVary('MedShkDstn')
        
    def updateIncomeProcess(self):
        '''
        Updates this agent's income process based on his own attributes.  The
        function that generates the discrete income process can be swapped out
        for a different process.
        
        Parameters
        ----------
        None
        
        Returns:
        -----------
        None
        '''
        original_time = self.time_flow
        self.timeFwd()
        
        # Rearrange some of the inputs
        PermShkStd    = map(list, zip(*self.PermShkStd))
        PermShkCount  = self.PermShkCount
        TranShkStd    = map(list, zip(*self.TranShkStd))
        TranShkCount  = self.TranShkCount
        T_cycle       = self.T_cycle
        T_retire      = self.T_retire
        UnempPrb      = self.UnempPrb
        IncUnemp      = self.IncUnemp
        UnempPrbRet   = self.UnempPrbRet        
        IncUnempRet   = self.IncUnempRet
        
        # Make empty nested lists to hold income distributions
        IncomeDstn = []
        TranShkDstn = []
        PermShkDstn = []
        for t in range(T_cycle):
            IncomeDstn.append([])
            TranShkDstn.append([])
            PermShkDstn.append([])
            
        # Loop through the discrete health states, adding income distributions for each one
        parameter_object = HARKobject()
        parameter_object(PermShkCount = PermShkCount,        
                         TranShkCount = TranShkCount,
                         T_cycle = T_cycle,
                         T_retire = T_retire,
                         UnempPrb = UnempPrb,
                         IncUnemp = IncUnemp,
                         UnempPrbRet = UnempPrbRet, 
                         IncUnempRet = IncUnempRet)
        for h in range(self.MrkvArray[0].shape[0]):
            parameter_object(PermShkStd=PermShkStd[h], TranShkStd=TranShkStd[h])
            IncomeDstn_temp, PermShkDstn_temp, TranShkDstn_temp = constructLognormalIncomeProcessUnemployment(parameter_object)
            for t in range(T_cycle):
                IncomeDstn[t].append(IncomeDstn_temp[t])
                PermShkDstn[t].append(PermShkDstn_temp[t])
                TranShkDstn[t].append(TranShkDstn_temp[t])
                
        # Strip out just the first Markov state's distributions, for use by updatePermIncGrid
        IncomeDstn_temp = [IncomeDstn[t][0] for t in range(T_cycle)]
        PermShkDstn_temp = [PermShkDstn[t][0] for t in range(T_cycle)]
        TranShkDstn_temp = [TranShkDstn[t][0] for t in range(T_cycle)]
            
        # Store the results as attributes of self
        self.IncomeDstn = IncomeDstn_temp
        self.PermShkDstn = PermShkDstn_temp
        self.TranShkDstn = TranShkDstn_temp
        self.IncomeDstn_all = IncomeDstn # These will be restored after updatePermIncGrid
        self.PermShkDstn_all = PermShkDstn
        self.TranShkDstn_all = TranShkDstn
        self.addToTimeVary('IncomeDstn','PermShkDstn','TranShkDstn')
        if not original_time:
            self.timeRev()
            
            
    def installPremiumFuncs(self):
        '''
        Applies the premium data in the attribute PremiumFuncs to the contracts
        in the attribute ContractList, accounting for the employer contribution
        in PremiumSubsidy.  The premiums in PremiumFuncs are used in the static
        solver to find quasi-equilibrium premiums; this method feeds these back
        into ContractList to use on the next dynamic solution pass.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        None
        '''
        if not hasattr(self,'PremiumFuncs'):
            return # Don't do anything if PremiumFuncs hasn't been defined
        time_orig = self.time_flow
        self.timeFwd()
        T_retire = 40 # For determining whether to apply subsidy
        T = len(self.PremiumFuncs)
        
        # Loop through each age-state-contract, filling in the updated premium in ContractList
        for t in range(T):
            StateCount = len(self.ContractList[t])
            for h in range(StateCount):
                ContractCount = len(self.ContractList[t][h])
                for z in range(ContractCount):
                    PremiumFunc = self.PremiumFuncs[t][h][z]
                    if PremiumFunc.__class__.__name__ == 'ConstantFunction':
                        NetPremium = np.maximum(PremiumFunc.value - (t < T_retire)*self.PremiumSubsidy, 0.0)
                        self.ContractList[t][h][z].Premium = ConstantFunction(NetPremium)
                    else:
                        raise TypeError("installPremiumFuncs can't yet handle non-ConstantFunctions!")
                        
        # Restore the original flow of time
        if not time_orig:
            self.timeRev()
            
            
    def makeMasterbFromxFuncs(self):
        '''
        For each effective price across all contracts that the individual will experience in his
        lifetime, construct a bFromxFunc that gives optimal consumption ratio as a function of total
        expenditure and the medical need shock.  Automatically chooses the grids of expenditure and
        medical need by looking at extreme outcomes over the individual's lifecycle.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        None
        '''
        pLvl_aug_factor = 6
        Med_aug_factor = 6
        
        # Make a master grid of expenditure levels
        pLvlGridAll = np.concatenate([pLvlGrid for pLvlGrid in self.pLvlGrid])
        pLvlGridAll = np.unique(pLvlGridAll)[1:] # remove duplicates and p=0
        pLogMin = np.log(pLvlGridAll[0])
        pLogMax = np.log(pLvlGridAll[-1])
        pLvlSet = np.exp(np.linspace(pLogMin,pLogMax,num=pLvl_aug_factor+1))
        pLvlSet = np.reshape(pLvlSet,(pLvlSet.size,1))
        xNrmSet = np.reshape(self.aXtraGrid,(1,self.aXtraGrid.size))
        xLvlGrid = np.unique((np.dot(pLvlSet,xNrmSet)).flatten())
        xLvlGrid = np.insert(xLvlGrid,0,0.0)
        
        # Make a master grid of medical need shocks
        MedShkAll = np.array([])
        for MedShkDstn in self.MedShkDstn:
            temp_list = [ThisDstn[1] for ThisDstn in MedShkDstn]
            MedShkAll = np.concatenate((MedShkAll,np.concatenate(temp_list)))
        MedShkAll = np.unique(MedShkAll)
        ShkLogMin = np.log(MedShkAll[1])
        ShkLogMax = np.log(MedShkAll[-1])
        MedShkCountTemp = self.MedShkCount
        if type(self.MedShkCountTail) is int:
            MedShkCountTemp += self.MedShkCountTail
        else:
            MedShkCountTemp += self.MedShkCountTail[0] + self.MedShkCountTail[1]
        MedShkGrid = np.insert(np.exp(np.linspace(ShkLogMin,ShkLogMax,num=Med_aug_factor*MedShkCountTemp)),0,0.0)
        
        # Make a master list of effective prices / coinsurance rates that the agent might ever experience
        PriceEffList = []
        for t in range(self.T_cycle):
            MedPrice = self.MedPrice[t]
            PriceEffList.append(MedPrice)
            for h in range(self.MrkvArray[0].shape[0]):
                for Contract in self.ContractList[t][h]:
                    PriceEffList.append(Contract.Copay*MedPrice)
        CopayListAll = np.unique(np.array(PriceEffList))
        
        # For each of those copays, make a bFromxFunc
        bFromxFuncListAll = []
        for MedPriceEff in CopayListAll:
            temp_object = MedShockPolicyFunc(NullFunc,xLvlGrid,MedShkGrid,MedPriceEff,self.CRRA,self.CRRAmed,xLvlCubicBool=self.CubicBool)
            bFromxFuncListAll.append(deepcopy(temp_object.bFunc))
            
        # Store the results in self
        self.CopayListAll = CopayListAll
        self.bFromxFuncListAll = bFromxFuncListAll
        
        
    def distributeConstructedFuncs(self):
        '''
        Constructs the attributes bFromxFuncList, CopayList, DfuncList, DpFuncList, cEffFuncList,
        GfuncList, and CritShkFuncList for each period in the agent's cycle. Should only be run
        after makeMasterbFromxFuncs() and makeMasterDfuncs(), which constructs the functions that
        will be reorganized here.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        None
        '''
        orig_time = self.time_flow
        self.timeFwd()
        
        # Initialize lists
        CopayList = [] # Coinsurance rates / effective medical care price for each period in the cycle
        bFromxFuncList = [] # Consumption ratio as function of expenditure and medical need for each period
        DfuncList = [] # Cost as a function of effective consumption and medical shock for each period
        DpFuncList = [] # Marginal cost as a function of effective consumption and medical shock for each period
        cEffFuncList = [] # Effective consumption as a function of total spending and medical shock for each period
        GfuncList = [] # Solution to first order condition as a function of pseudo inverse end-of-period marginal value and medical shock for each period
        CritShkFuncList = [] # Medical need shock above which consumption floor binds, as function of total expenditure

        # Loop through time and construct the lists above
        for t in range(self.T_cycle):
            MedPrice = self.MedPrice[t]
            CopayList_temp = [MedPrice]
            for h in range(len(self.ContractList[t])):
                for Contract in self.ContractList[t][h]:
                    CopayList_temp.append(Contract.Copay*MedPrice)
            CopayList.append(np.unique(np.array(CopayList_temp)))
            bFromxFuncList_temp = []
            DfuncList_temp = []
            DpFuncList_temp = []
            cEffFuncList_temp = []
            GfuncList_temp = []
            CritShkFuncList_temp = []
            for Copay in CopayList[-1]:
                # For each copay, add the appropriate function to each list
                idx = np.argwhere(np.array(self.CopayListAll)==Copay)[0][0]
                bFromxFuncList_temp.append(self.bFromxFuncListAll[idx])
                DfuncList_temp.append(self.DfuncListAll[idx])
                DpFuncList_temp.append(self.DpFuncListAll[idx])
                cEffFuncList_temp.append(self.cEffFuncListAll[idx])
                GfuncList_temp.append(self.GfuncListAll[idx])
                CritShkFuncList_temp.append(self.CritShkFuncListAll[idx])
                
            # Add the one-period lists to the full list
            bFromxFuncList.append(copy(bFromxFuncList_temp))
            DfuncList.append(copy(DfuncList_temp))
            DpFuncList.append(copy(DpFuncList_temp))
            cEffFuncList.append(copy(cEffFuncList_temp))
            GfuncList.append(copy(GfuncList_temp))
            CritShkFuncList.append(copy(CritShkFuncList_temp))
            
        # Store the results in self, add to time_vary, and restore time to its original direction
        self.CopayList = CopayList
        self.bFromxFuncList = bFromxFuncList
        self.DfuncList = DfuncList
        self.DpFuncList = DpFuncList
        self.cEffFuncList = cEffFuncList
        self.GfuncList = GfuncList
        self.CritShkFuncList = CritShkFuncList
        self.addToTimeVary('CopayList','bFromxFuncList','DfuncList','DpFuncList','cEffFuncList','GfuncList','CritShkFuncList')
        if not orig_time:
            self.timeRev()
            
            
    def makeMasterDfuncs(self):
        '''
        For each effective price across all contracts that the individual will experience in his
        lifetime, construct a Dfunc that gives the dollar cost of attaining "effective consumption"
        C when facing medical need shock MedShk: D = Dfunc(C,eta).  Can only be executed after
        running the method makeMasterbFromxFuncs(), as the construction of Dfunc and DpFunc use
        the solution to the optimal spending composition problem solved there.  Also makes
        cEffFunc for each effective price, mapping expenditure levels and medical shocks to
        maximum effective consumption levels achievable.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        None
        '''
        K = self.CopayListAll.size # These are really EffMedPrice values
        DfuncList = []
        DpFuncList = []
        cEffFuncList = []
        GfuncList = []
        CritShkFuncList = []
        
        # Loops through the effective medical prices and construct a cost function for each
        for k in range(K):
            # Unpack basics from the bFromxFunc
            MedPriceEff = self.CopayListAll[k]
            bFromxFunc = self.bFromxFuncListAll[k]
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
            cEffGrid = (1.-np.exp(-MedGrid/ShkGrid))**self.CRRAmed*cGrid
            cEffGrid[:,0] = xVec

            # Construct a grid of marginal effective consumption values
            temp1 = np.exp(-MedGrid/ShkGrid)
            temp2 = np.exp(MedGrid/ShkGrid)
            dcdx = temp2/(temp2 + self.CRRAmed)
            dcdx[ShkZero] = 1.
            dMeddx = (1./MedPriceEff)*self.CRRAmed/(temp2 + self.CRRAmed)
            dMeddx[ShkZero] = 0.
            dcEffdx = self.CRRAmed/ShkGrid*temp1*dMeddx*(1.-temp1)**(self.CRRAmed-1.)*cGrid + (1.-temp1)**self.CRRAmed*dcdx
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
            
            # For each log(x) value, find the critical shock where the consumption floor binds
            N = LogcEffGrid.shape[0]
            LogCfloor = np.log(self.Cfloor)
            IdxVec = np.maximum(np.minimum(np.sum(LogcEffGrid >= LogCfloor,axis=1),ShkVec.size-1),1)
            LogC1 = LogcEffGrid[np.arange(N),IdxVec]
            LogC0 = LogcEffGrid[np.arange(N),IdxVec-1]
            alpha = (LogCfloor-LogC0)/(LogC1-LogC0)
            CritShk = np.maximum((1.-alpha)*ShkVec[IdxVec-1] + alpha*ShkVec[IdxVec],0.)
            i = np.sum(CritShk==0.)
            CritShkFuncList.append(LinearInterp(np.insert(xVec[1:],i,self.Cfloor),np.insert(CritShk,i,0.0),lower_extrap=True))
            
        # Store these functions as attributes of the agent
        self.DfuncListAll = DfuncList
        self.DpFuncListAll = DpFuncList
        self.cEffFuncListAll = cEffFuncList
        self.GfuncListAll = GfuncList
        self.CritShkFuncListAll = CritShkFuncList
            
    
    def updateSolutionTerminal(self):
        '''
        Solve for the terminal period solution.  For the sake of simplicity, assume that the last
        non-terminal period's data is also used for the terminal period (this can be changed later).
        
        Parameters
        ----------
        None
        
        Returns
        -------
        None
        '''
        t_start = clock()
        
        # Define utility function derivatives and inverses
        CRRA = self.CRRA
        CRRAmed = self.CRRAmed
        u = lambda x : utility(x,gam=CRRA)
        #uP = lambda x : utilityP(x,gam=CRRA)
        uPinv = lambda x : utilityP_inv(x,gam=CRRA)
        uinvP = lambda x : utility_invP(x,gam=CRRA)
        uinv = lambda x : utility_inv(x,gam=CRRA)
        uPinvP = lambda x : utilityP_invP(x,gam=CRRA)
        
        # Take last period data, whichever way time is flowing
        if self.time_flow:
            t = -1
        else:
            t = 0
        MedPrice = self.MedPrice[t]
        ContractList = self.ContractList[t]
        CopayList = self.CopayList[t]
        bFromxFuncList = self.bFromxFuncList[t]
        pLvlGrid = self.pLvlGrid[t]
        ChoiceShkMag = self.ChoiceShkMag[t]
        CritShkFuncList = self.CritShkFuncList[t]
        DpFuncList = self.DpFuncList[t]
        cEffFuncList = self.cEffFuncList[t]
        MedShkAvg = self.MedShkAvg[t]
        MedShkStd = self.MedShkStd[t]
        MedShkDstn = self.MedShkDstn[t]
        
        # Make the expenditure function for the terminal period
        xFunc_terminal = IdentityFunction(i_dim=0,n_dims=3)
            
        # Make grids for the three state dimensions
        MedShkGrid= bFromxFuncList[0].y_list
        pLvlCount = pLvlGrid.size
        aLvlCount = self.aXtraGrid.size
        ShkCount  = MedShkGrid.size
        pLvlNow   = np.tile(pLvlGrid,(aLvlCount,1)).transpose()
        mLvlNow   = np.tile(self.aXtraGrid,(pLvlCount,1))*pLvlNow
        pLvlNow_tiled = np.tile(pLvlNow,(ShkCount,1,1))
        mLvlNow_tiled = np.tile(mLvlNow,(ShkCount,1,1)) # shape = (ShkCount,pLvlCount,aLvlCount)
        if pLvlGrid[0] == 0.0:  # aLvl turns out badly if pLvl is 0 at bottom
            mLvlNow[0,:] = self.aXtraGrid*pLvlGrid[1]
            mLvlNow_tiled[:,0,:] = np.tile(self.aXtraGrid*pLvlGrid[1],(ShkCount,1))
        MedShkVals_tiled  = np.transpose(np.tile(MedShkGrid,(aLvlCount,pLvlCount,1)),(2,1,0))
        
        # Construct policy and value functions for each coinsurance rate
        policyFuncs_by_copay = []
        vFuncs_by_copay = []
        for j in range(len(CopayList)):
            Copay = CopayList[j]
            policyFuncs_by_copay.append(MedShockPolicyFuncPrecalc(xFunc_terminal,bFromxFuncList[j],Copay))
            xLvlNow = xFunc_terminal(mLvlNow_tiled,pLvlNow_tiled,MedShkVals_tiled)
            cEffLvlNow = cEffFuncList[j](xLvlNow,MedShkVals_tiled)
            vNow = u(cEffLvlNow)
            #vPnow = uP(cLvlNow)
            vNvrsNow  = np.concatenate((np.zeros((ShkCount,pLvlCount,1)),uinv(vNow)),axis=2)
            #vNvrsPnow = np.concatenate((np.zeros((ShkCount,pLvlCount,1)),vPnow*uinvP(vNow)),axis=2)
            mLvlNow_adj = np.concatenate((np.tile(np.reshape(np.zeros(pLvlCount),(1,pLvlCount,1)),(ShkCount,1,1)),mLvlNow_tiled),axis=2)
            vFuncNvrs_by_pLvl_and_MedShk = [] # Initialize the empty list of lists of 1D vFuncs
            for i in range(pLvlCount):
                temp_list = []
                for j in range(ShkCount):
                    m_temp = mLvlNow_adj[j,i,:]
                    vNvrs_temp = vNvrsNow[j,i,:]
                    #vNvrsPnow_temp = vNvrsPnow[j,i,:]
                    #temp_list.append(CubicInterp(m_temp,vNvrs_temp,vNvrsPnow_temp))
                    temp_list.append(LinearInterp(m_temp,vNvrs_temp))
                vFuncNvrs_by_pLvl_and_MedShk.append(deepcopy(temp_list))
            vFuncNvrs = BilinearInterpOnInterp1D(vFuncNvrs_by_pLvl_and_MedShk,pLvlGrid,MedShkGrid)
            vFuncs_by_copay.append(ValueFunc3D(vFuncNvrs,CRRA))
            
        
        # Loop through each health state and solve the terminal period
        solution_terminal = InsuranceSelectionSolution(mLvlMin=ConstantFunction(0.0))
        MedShkCritArrayList = []
        vArrayBigList = []
        for h in range(self.MrkvArray[0].shape[0]):
            # Set up state grids to prepare for the "integration" step
            MedShkVals = MedShkDstn[h][1]
            MedShkPrbs = MedShkDstn[h][0]
            MedCount = MedShkVals.size
            ZgridBase = (np.log(MedShkVals) - MedShkAvg[h])/MedShkStd[h] # Find baseline Z-grids for the medical shock distribution
            tempArray    = np.tile(np.reshape(self.aXtraGrid,(aLvlCount,1,1)),(1,pLvlCount,MedCount))
            mMinArray    = np.zeros((aLvlCount,pLvlCount,MedCount))
            pLvlArray    = np.tile(np.reshape(pLvlGrid,(1,pLvlCount,1)),(aLvlCount,1,MedCount))
            mLvlArray    = mMinArray + tempArray*pLvlArray + self.Cfloor
            if pLvlGrid[0] == 0.0: # Fix the problem of all mLvls = 0 when pLvl = 0
                mLvlArray[:,0,:] = np.tile(np.reshape(self.aXtraGrid*pLvlGrid[1],(aLvlCount,1)),(1,MedCount)) + self.Cfloor         
            MedShkArray  = np.tile(np.reshape(MedShkVals,(1,1,MedCount)),(aLvlCount,pLvlCount,1))
            
            # For each insurance contract available in this health state, "integrate" across medical need
            # shocks to get policy and (marginal) value functions for each contract
            policyFuncsThisHealth = []
            vFuncsThisHealth = []
            vPfuncsThisHealth = []
            for z in range(len(ContractList[h])):
                # Set and unpack the contract of interest
                Contract = ContractList[h][z]
                Copay = Contract.Copay
                FullPrice_idx = np.argwhere(np.array(CopayList)==MedPrice)[0][0]
                Copay_idx = np.argwhere(np.array(CopayList)==(Copay*MedPrice))[0][0]
               
                # Get the value and policy functions for this contract
                vFuncFullPrice = vFuncs_by_copay[FullPrice_idx]
                vFuncCopay = vFuncs_by_copay[Copay_idx]
                policyFuncFullPrice = policyFuncs_by_copay[FullPrice_idx]
                policyFuncCopay = policyFuncs_by_copay[Copay_idx]
                DpFuncFullPrice = DpFuncList[FullPrice_idx]
                DpFuncCopay = DpFuncList[Copay_idx]
            
                # Make the policy function for this contract and find critical medical need shocks
                policyFuncsThisHealth.append(InsSelPolicyFunc(vFuncFullPrice,vFuncCopay,policyFuncFullPrice,policyFuncCopay,DpFuncFullPrice,DpFuncCopay,Contract,CRRAmed))
                MedShkCritArray = CritShkFuncList[Copay_idx](mLvlArray[:,:,0]) # Assume zero deductible
                these = MedShkCritArray > 0.
                MedShkCritArrayList.append(MedShkCritArray)
                
                # Translate critical MedShk into array of shocks and probabilities to use in integration
                AlwaysCfloor = np.logical_not(these)
                ZcritArray = (np.log(MedShkCritArray) - MedShkAvg[h])/MedShkStd[h]
                CfloorPrbArray = norm.sf(ZcritArray)*(1.-MedShkPrbs[0])
                CfloorPrbArray[AlwaysCfloor] = 1.0 # These were shifted down by line above, but that's wrong
                ZadjArray = np.minimum(ZcritArray - ZgridBase[-1],0.) # Should always be non-positive
                ZshkArray = np.tile(np.reshape(ZgridBase,(1,1,ZgridBase.size)),(aLvlCount,pLvlCount,1)) + np.tile(np.reshape(ZadjArray,(aLvlCount,pLvlCount,1)),(1,1,MedCount))
                MedShkArray = np.exp(ZshkArray*MedShkStd[h] + MedShkAvg[h])
                TempPrbArray = norm.pdf(ZshkArray)
                ReweightArray = (1.-MedShkPrbs[0]-CfloorPrbArray)/np.sum(TempPrbArray,axis=2)
                ShkPrbsArray = TempPrbArray*np.tile(np.reshape(ReweightArray,(aLvlCount,pLvlCount,1)),(1,1,MedCount))
                ShkPrbsArray[:,:,0] = MedShkPrbs[0]
                AlwaysCfloor_tiled = np.tile(np.reshape(AlwaysCfloor,(aLvlCount,pLvlCount,1)),(1,1,MedCount))
                ShkPrbsArray[AlwaysCfloor_tiled] = 0.0
                            
                # Get value and marginal value at an array of states and integrate across medical shocks
                vArrayBig, vParrayBig, vPParrayBig = policyFuncsThisHealth[-1].evalvAndvPandvPP(mLvlArray,pLvlArray,MedShkArray)
                vArrayBigList.append(vArrayBig)
                vArray   = np.sum(vArrayBig*ShkPrbsArray,axis=2) + u(self.Cfloor)*CfloorPrbArray
                vParray  = np.sum(vParrayBig*ShkPrbsArray,axis=2)
                if self.CubicBool:
                    vPParray = np.sum(vPParrayBig*ShkPrbsArray,axis=2)
                    
                # Make pseudo-inverse versions of value and marginal value arrays
                vNvrsFloor = self.Cfloor
                mLvlBound = mLvlArray[0,:,0] # Lowest mLvl for each pLvl; will be used to make "seam"
                mGrid_small_A = np.concatenate((np.zeros((1,pLvlCount)),self.Cfloor*np.ones((1,pLvlCount)),mLvlArray[:,:,0]),axis=0)
                mGrid_small_B = mLvlArray[:,:,0] - np.tile(np.reshape(mLvlBound,(1,pLvlCount)),(aLvlCount,1))
                vPnvrsArray   = uPinv(vParray)
                vNvrsArray    = np.concatenate((vNvrsFloor*np.ones((2,pLvlCount)),uinv(vArray)),axis=0)
                if self.CubicBool:
                    vPnvrsParray  = np.concatenate((np.zeros((1,pLvlCount)),vPParray*uPinvP(vParray)),axis=0)
                
                # Construct the pseudo-inverse value and marginal value functions over mLvl,pLvl
                vPnvrsFuncUpper_by_pLvl = []
                vNvrsFunc_by_pLvl = []
                vPfuncLower_by_pLvl = []
                for j in range(pLvlCount): # Make a pseudo inverse marginal value function for each pLvl
                    vPfuncLower_by_pLvl.append(LinearInterp(np.array([0.,self.Cfloor,mLvlBound[j]]),np.array([0.,0.,vParray[0,j]])))
                    m_temp_A = mGrid_small_A[:,j]
                    m_temp_B = mGrid_small_B[:,j]
                    vPnvrs_temp = vPnvrsArray[:,j]
                    vPnvrsFuncUpper_by_pLvl.append(LinearInterp(m_temp_B,vPnvrs_temp))
                    vNvrs_temp  = vNvrsArray[:,j]
                    vNvrsFunc_by_pLvl.append(LinearInterp(m_temp_A,vNvrs_temp))
                vNvrsFunc  = LinearInterpOnInterp1D(vNvrsFunc_by_pLvl,pLvlGrid)
                vFuncThisContract = ValueFunc2D(vNvrsFunc,CRRA) # Recurve the value function
                  
                # Construct the marginal value function as a composite function
                vPfuncSeam = LinearInterp(pLvlGrid,mLvlBound)
                vPnvrsFuncUpperBase = LinearInterpOnInterp1D(vPnvrsFuncUpper_by_pLvl,pLvlGrid)
                vPnvrsFuncUpper = VariableLowerBoundFunc2D(vPnvrsFuncUpperBase,vPfuncSeam)
                vPfuncUpper = MargValueFunc2D(vPnvrsFuncUpper,CRRA) # "Re-curve" the upper marginal value function
                vPfuncLower = LinearInterpOnInterp1D(vPfuncLower_by_pLvl,pLvlGrid)
                vPfuncThisContract = CompositeFunc2D(vPfuncLower,vPfuncUpper,vPfuncSeam)
                
                # Store the policy and (marginal) value function
                vFuncsThisHealth.append(vFuncThisContract)
                vPfuncsThisHealth.append(vPfuncThisContract)
            
            # Do choice shock step iff there is more than one contract to choose from
            if len(ContractList[h]) == 1: # Skip choice shock step if only one contract
                vFunc = vFuncsThisHealth[0]
                vPfunc = vPfuncsThisHealth[0]
            else:   
                # Make grids to prepare for the choice shock step
                tempArray    = np.tile(np.reshape(self.aXtraGrid,(aLvlCount,1)),(1,pLvlCount))
                mMinArray    = np.tile(np.reshape(np.zeros(pLvlCount),(1,pLvlCount)),(aLvlCount,1))
                pLvlArray    = np.tile(np.reshape(pLvlGrid,(1,pLvlCount)),(aLvlCount,1))
                mLvlArray    = mMinArray + tempArray*pLvlArray
                if pLvlGrid[0] == 0.0: # Fix the problem of all mLvls = 0 when pLvl = 0
                    mLvlArray[:,0] = self.aXtraGrid
                
                # Get value and marginal value at each point in mLvl x pLvl at each contract, taking account of premiums paid
                vArrayBig   = np.zeros((aLvlCount,pLvlCount,len(ContractList[h]))) # value at each gridpoint for each contract
                vParrayBig  = np.zeros((aLvlCount,pLvlCount,len(ContractList[h]))) # marg value at each gridpoint for each contract
                vPParrayBig = np.zeros((aLvlCount,pLvlCount,len(ContractList[h]))) # marg marg value at each gridpoint for each contract
                AdjusterArray = np.zeros((aLvlCount,pLvlCount,len(ContractList[h]))) # (1 - dPremium/dmLvl)
                UnaffordableArray = np.zeros((aLvlCount,pLvlCount,len(ContractList[h])),dtype=bool)
                for z in range(len(ContractList[h])):
                    Contract = ContractList[h][z]
                    PremiumArray = Contract.Premium(mLvlArray,pLvlArray)
                    AdjusterArray[:,:,z] = 1.0 - Contract.Premium.derivativeX(mLvlArray,pLvlArray)
                    mLvlArray_temp = mLvlArray-PremiumArray
                    UnaffordableArray[:,:,z] = mLvlArray_temp <= 0
                    vArrayBig[:,:,z]   = vFuncsThisHealth[z](mLvlArray_temp,pLvlArray)
                    vParrayBig[:,:,z]  = vPfuncsThisHealth[z](mLvlArray_temp,pLvlArray)
                    if self.CubicBool:
                        vPParrayBig[:,:,z] = vPfuncsThisHealth[z].derivativeX(mLvlArray_temp,pLvlArray)
                          
                # Transform value (etc) into the pseudo-inverse forms needed
                vNvrsArrayBig  = uinv(vArrayBig)
                vNvrsParrayBig = AdjusterArray*vParrayBig*uinvP(vArrayBig)
                vPnvrsArrayBig = uPinv(AdjusterArray*vParrayBig)
                if self.CubicBool:
                    Temp = np.zeros_like(AdjusterArray)
                    for z in range(len(ContractList[h])):
                        Contract = ContractList[h][z]
                        Temp[:,:,z] = Contract.Premium.derivativeXX(mLvlArray,pLvlArray)
                    vPnvrsParrayBig = (AdjusterArray*vPParrayBig - Temp*vParrayBig)*uPinvP(AdjusterArray*vParrayBig)
                
                # Fix the unaffordable points so they don't generate NaNs near bottom
                vNvrsArrayBig[UnaffordableArray] = -np.inf
                vNvrsParrayBig[UnaffordableArray] = 0.0
                vPnvrsArrayBig[UnaffordableArray] = 0.0
                if self.CubicBool:
                    vPnvrsParrayBig[UnaffordableArray] = 0.0
                
                # Weight each gridpoint by its contract probabilities
                if ChoiceShkMag < 0.0: # Never use choice shocks in solution
                    v_best = np.max(vNvrsArrayBig,axis=2)
                    v_best_tiled = np.tile(np.reshape(v_best,(aLvlCount,pLvlCount,1)),(1,1,len(ContractList[h])))
                    vNvrsArrayBig_adjexp = np.exp((vNvrsArrayBig - v_best_tiled)/ChoiceShkMag)
                    vNvrsArrayBig_adjexpsum = np.sum(vNvrsArrayBig_adjexp,axis=2)
                    vNvrsArray = ChoiceShkMag*np.log(vNvrsArrayBig_adjexpsum) + v_best
                    ContractPrbs = vNvrsArrayBig_adjexp/np.tile(np.reshape(vNvrsArrayBig_adjexpsum,(aLvlCount,pLvlCount,1)),(1,1,len(ContractList[h])))
                    #vNvrsParray = np.sum(ContractPrbs*vNvrsParrayBig,axis=2)
                    vPnvrsArray = np.sum(ContractPrbs*vPnvrsArrayBig,axis=2)
                    if self.CubicBool:
                        vPnvrsParray = np.sum(ContractPrbs*vPnvrsParrayBig,axis=2)
                else:
                    z_choice = np.argmax(vNvrsArrayBig,axis=2)
                    vNvrsArray = np.zeros((aLvlCount,pLvlCount))
                    vPnvrsArray = np.zeros((aLvlCount,pLvlCount))
                    for z in range(len(ContractList[h])):
                        these = z_choice == z
                        vNvrsArray[these] = vNvrsArrayBig[:,:,z][these]
                        vPnvrsArray[these] = vPnvrsArrayBig[:,:,z][these]
                    
                # Make value and marginal value functions for the very beginning of the period, before choice shocks are drawn
                vNvrsFuncs_by_pLvl = []
                vPnvrsFuncs_by_pLvl = []
                for j in range(pLvlCount): # Make 1D functions by pLvl
                    m_temp = np.insert(mLvlArray[:,j],0,0.0)
                    vNvrs_temp   = np.insert(vNvrsArray[:,j],0,vNvrsFloor)
                    #vNvrsP_temp  = np.insert(vNvrsParray[:,j],0,vNvrsParray[0,j])
                    #vNvrsFuncs_by_pLvl.append(CubicInterp(m_temp,vNvrs_temp,vNvrsP_temp))
                    vNvrsFuncs_by_pLvl.append(LinearInterp(m_temp,vNvrs_temp))
                    vPnvrs_temp  = np.insert(vPnvrsArray[:,j],0,0.0)
                    if self.CubicBool:
                        vPnvrsP_temp = np.insert(vPnvrsParray[:,j],0,vPnvrsParray[0,j])
                        vPnvrsFuncs_by_pLvl.append(CubicInterp(m_temp,vPnvrs_temp,vPnvrsP_temp))
                    else:
                        vPnvrsFuncs_by_pLvl.append(LinearInterp(m_temp,vPnvrs_temp))
                
                # Combine across pLvls and re-curve the functions
                vPnvrsFunc = LinearInterpOnInterp1D(vPnvrsFuncs_by_pLvl,pLvlGrid)
                vPfunc     = MargValueFunc2D(vPnvrsFunc,CRRA)
                vNvrsFunc  = LinearInterpOnInterp1D(vNvrsFuncs_by_pLvl,pLvlGrid)
                vFunc      = ValueFunc2D(vNvrsFunc,CRRA)
                if self.CubicBool:
                    vPPfunc    = MargMargValueFunc2D(vPnvrsFunc,CRRA)
                
            # Make the human wealth function for this health state and add solution to output
            hLvl_h = LinearInterp(np.insert(pLvlGrid,0,0.0),np.zeros(pLvlCount+1))
            if self.CubicBool:
                solution_terminal.appendSolution(vFunc=vFunc,vPfunc=vPfunc,vPPfunc=vPPfunc,hLvl=hLvl_h,
                                        policyFunc=policyFuncsThisHealth,vFuncByContract=vFuncsThisHealth)
            else:
                solution_terminal.appendSolution(vFunc=vFunc,vPfunc=vPfunc,hLvl=hLvl_h,
                                        policyFunc=policyFuncsThisHealth,vFuncByContract=vFuncsThisHealth)
        
        # Store the terminal period solution in self
        solution_terminal.MedShkCritArrayList = MedShkCritArrayList
        solution_terminal.vArrayBigList = vArrayBigList
        self.solution_terminal = solution_terminal
        t_end = clock()
        #print('Solving terminal period took ' + str(t_end-t_start) + ' seconds.')
                
        
    def update(self):
        '''
        Make constructed inputs for solving the model.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        None
        '''
        self.updateAssetsGrid()
        
        self.PermGroFac_all = self.PermGroFac
        self.PermGroFac = [self.PermGroFac[t][0] for t in range(self.T_cycle)]
        self.updatepLvlNextFunc() # Use only one Markov state's data temporarily
        self.PermGroFac = self.PermGroFac_all
        
        self.updateIncomeProcess()
        self.updatePermIncGrid()
        self.IncomeDstn = self.IncomeDstn_all # these attributes temporarily only had one Markov state's data
        self.PermShkDstn = self.PermShkDstn_all
        self.TranShkDstn = self.TranShkDstn_all
        
        self.updateMedShockProcess()
        self.makeMasterbFromxFuncs()
        self.makeMasterDfuncs()
        self.distributeConstructedFuncs()
        
    def preSolve(self):
        self.update()
        
        
    def makeShockHistory(self):
        '''
        Makes complete histories of health states, permanent income levels,
        medical needs shocks, labor income, and mortality for all agents.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        None
        '''
        # Make initial health state (Markov states)
        N = self.AgentCount
        base_draws = drawUniform(N,seed=self.RNG.randint(0,2**31-1))
        Cutoffs = np.cumsum(np.array(self.MrkvPrbsInit))
        MrkvNow = np.searchsorted(Cutoffs,base_draws).astype(int)
        
        # Make initial permanent income and asset levels, etc
        pLvlInit = np.exp(self.pLvlInitMean)*drawMeanOneLognormal(N=self.AgentCount,sigma=self.pLvlInitStd,seed=self.RNG.randint(0,2**31-1))
        self.aLvlInit = 0.3*pLvlInit
        pLvlNow = pLvlInit
        Live = np.ones(self.AgentCount,dtype=bool)
        Dead = np.logical_not(Live)
        PermShkNow = np.zeros_like(pLvlInit)
        TranShkNow = np.ones_like(pLvlInit)
        MedShkNow  = np.zeros_like(pLvlInit)
        
        # Make blank histories for the state and shock variables
        T_sim = self.T_sim
        N_agents = N
        pLvlHist = np.zeros((T_sim,N_agents)) + np.nan
        MrkvHist = -np.ones((T_sim,N_agents),dtype=int)
        PrefShkHist = np.reshape(drawUniform(N=T_sim*N_agents,seed=self.RNG.randint(0,2**31-1)),(T_sim,N_agents))
        MedShkHist = np.zeros((T_sim,N_agents)) + np.nan
        IncomeHist = np.zeros((T_sim,N_agents)) + np.nan

        # Loop through each period of life and update histories
        t_cycle=0
        for t in range(T_sim):
            # Add current states to histories
            pLvlHist[t,Live] = pLvlNow[Live]
            MrkvHist[t,Live] = MrkvNow[Live]
            IncomeHist[t,:] = pLvlNow*TranShkNow

            # Get income and medical shocks for next period in each health state
            PermShkNow[:] = np.nan
            TranShkNow[:] = np.nan
            MedShkNow[:] = np.nan
            for h in range(5):
                these = MrkvNow == h
                N = np.sum(these)
                
                # First income shocks for next period...
                IncDstn_temp = self.IncomeDstn[t_cycle][h]
                probs = IncDstn_temp[0]
                events = np.arange(probs.size)
                idx = drawDiscrete(N=N,P=probs,X=events,seed=self.RNG.randint(0,2**31-1)).astype(int)
                PermShkNow[these] = IncDstn_temp[1][idx]
                TranShkNow[these] = IncDstn_temp[2][idx]

                # Then medical needs shocks for this period...
                sigma = self.MedShkStd[t_cycle][h]
                mu = self.MedShkAvg[t_cycle][h]
                MedShkDraws = drawLognormal(N=N,mu=mu,sigma=sigma,seed=self.RNG.randint(0,2**31-1))
                zero_prb = self.ZeroMedShkPrb[t_cycle,h]
                zero_med = drawBernoulli(N=N,p=zero_prb,seed=self.RNG.randint(0,2**31-1))
                MedShkDraws[zero_med] = 0.0
                MedShkNow[these] = MedShkDraws
                
            # Store the medical shocks and update permanent income for next period
            MedShkHist[t,:] = MedShkNow
            pLvlNow = self.pLvlNextFunc[t](pLvlNow)*PermShkNow
            
            # Determine which agents die based on their mortality probability
            LivPrb_temp = self.LivPrb[t_cycle][MrkvNow[Live]]
            LivPrbAll = np.zeros_like(pLvlNow)
            LivPrbAll[Live] = LivPrb_temp
            MortShkNow = drawUniform(N=self.AgentCount,seed=self.RNG.randint(0,2**31-1))
            Dead = MortShkNow > LivPrbAll
            Live = np.logical_not(Dead)
            
            # Draw health states for survivors next period
            events = np.arange(5)
            MrkvNext = MrkvNow
            for h in range(5):
                these = MrkvNow == h
                N = np.sum(these)
                probs = self.MrkvArray[t_cycle][h,:]
                idx = drawDiscrete(N=N,P=probs,X=events,seed=self.RNG.randint(0,2**31-1)).astype(int)
                MrkvNext[these] = idx
            MrkvNext[Dead] = -1 # Actually kill those who died
            MrkvNow = MrkvNext  # Next period will soon be this period
            
            # Advance t_cycle or reset it to zero
            t_cycle += 1
            if t_cycle == self.T_cycle:
                t_cycle = 0
                
        # Make boolean arrays for health state for all agents
        HealthBoolArray = np.zeros((self.T_sim,self.AgentCount,5),dtype=bool)
        for h in range(5):
            HealthBoolArray[:,:,h] = MrkvHist == h
        self.HealthBoolArray = HealthBoolArray
        self.LiveBoolArray = np.any(HealthBoolArray,axis=2)
            
        # Store the history arrays as attributes of self
        self.pLvlHist = pLvlHist
        self.IncomeHist = IncomeHist
        self.MrkvHist = MrkvHist
        self.MedShkHist = MedShkHist
        self.PrefShkHist = PrefShkHist

    def simBirth(self,which_agents):
        '''
        Very simple method for initializing agents.  Only called at the very
        beginning of simulation because this type uses "history style".
        '''
        self.aLvlNow[which_agents] = self.aLvlInit[which_agents]

    def marketAction(self):
        '''
        Calculates total expected medical costs by age-state-contract, as well
        as the number of agents who buy the contract.
        '''
        self.calcExpInsPayByContract()
        
    def simOnePeriod(self):
        '''
        Simulates one period of the insurance selection model.  The simulator
        uses the "history" style of simulation and should only be run after
        executing both the solve() and makeShockHistory() methods.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        None
        '''
        t = self.t_sim
        MedPriceNow = self.MedPrice[t]
        CeffFunc = lambda c,Med,Shk : (1.-np.exp(-Med/Shk))**self.CRRAmed*c
        FullPrice_idx = np.argwhere(self.CopayList[t]==MedPriceNow)[0][0]
        CfloorCostFunc = lambda Shk : self.DfuncList[t][FullPrice_idx](self.Cfloor*np.ones_like(Shk),Shk)
        FullPricebFunc = self.bFromxFuncList[t][FullPrice_idx]
        
        
        # Get state and shock vectors for (living) agents
        mLvlNow = self.aLvlNow + self.IncomeHist[t,:]
        HealthNow = self.MrkvHist[t,:]
        pLvlNow = self.pLvlHist[t,:]
        MedShkNow = self.MedShkHist[t,:]
        PrefShkNow = self.PrefShkHist[t,:]

        # Loop through each health state and get agents' controls
        cLvlNow = np.zeros_like(mLvlNow) + np.nan
        OOPnow = np.zeros_like(mLvlNow) + np.nan
        MedLvlNow = np.zeros_like(mLvlNow) + np.nan
        xLvlNow = np.zeros_like(mLvlNow) + np.nan
        PremNow = np.zeros_like(mLvlNow) + np.nan
        ContractNow = -np.ones(self.AgentCount,dtype=int)
        for h in range(5):
            these = HealthNow == h
            N = np.sum(these)
            Z = len(self.solution[t].policyFunc[h])
            
            # Get the pseudo-inverse value of holding each contract
            vNvrs_temp = np.zeros((N,Z)) + np.nan
            m_temp = mLvlNow[these]
            p_temp = pLvlNow[these]
            for z in range(Z):                
                Premium = self.solution[t].policyFunc[h][z].Contract.Premium(m_temp,p_temp)
                vNvrs_temp[:,z] = self.solution[t].vFuncByContract[h][z].func(m_temp-Premium,p_temp)
            
            if self.ChoiceShkMag[t] > 0.:
                # Get choice probabilities for each contract
                vNvrs_temp[np.isnan(vNvrs_temp)] = -np.inf
                v_best = np.max(vNvrs_temp,axis=1)
                v_best_big = np.tile(np.reshape(v_best,(N,1)),(1,Z))
                v_adj_exp = np.exp((vNvrs_temp - v_best_big)/self.ChoiceShkMag[t])
                v_sum_rep = np.tile(np.reshape(np.sum(v_adj_exp,axis=1),(N,1)),(1,Z))
                ChoicePrbs = v_adj_exp/v_sum_rep
                Cutoffs = np.cumsum(ChoicePrbs,axis=1)
                
                # Select a contract for each agent based on the unified preference shock
                PrefShk_temp = np.tile(np.reshape(PrefShkNow[these],(N,1)),(1,Z))
                z_choice = np.sum(PrefShk_temp > Cutoffs,axis=1).astype(int)
            else:
                z_choice = np.argmax(vNvrs_temp,axis=1) # Just choose best one if there are no shocks
            
            # For each contract, get controls for agents who buy it
            c_temp = np.zeros(N) + np.nan
            Med_temp = np.zeros(N) + np.nan
            OOP_temp = np.zeros(N) + np.nan
            Prem_temp = np.zeros(N) + np.nan
            MedShk_temp = MedShkNow[these]
            for z in range(Z):
                idx = z_choice == z
                Prem_temp[idx] = self.solution[t].policyFunc[h][z].Contract.Premium(m_temp[idx],p_temp[idx])
                c_temp[idx],Med_temp[idx] = self.solution[t].policyFunc[h][z](m_temp[idx]-Prem_temp[idx],p_temp[idx],MedShk_temp[idx])
                Med_temp[idx] = np.maximum(Med_temp[idx],0.0) # Prevents numeric glitching
                OOP_temp[idx] = self.solution[t].policyFunc[h][z].Contract.OOPfunc(Med_temp[idx])            
            
            # Store the controls for this health
            cLvlNow[these] = c_temp
            MedLvlNow[these] = Med_temp
            xLvlNow[these] = c_temp + OOP_temp
            OOPnow[these] = OOP_temp
            PremNow[these] = Prem_temp
            ContractNow[these] = z_choice
        aLvlNow = mLvlNow - PremNow - xLvlNow
            
        # Handle the consumption floor
        Ceff = CeffFunc(cLvlNow,MedLvlNow,MedShkNow)
        NeedHelp = Ceff < self.Cfloor
        CfloorMedShk = MedShkNow[NeedHelp]
        CfloorxLvl = CfloorCostFunc(CfloorMedShk)
        Welfare = CfloorxLvl - mLvlNow[NeedHelp]
        b_temp = FullPricebFunc(CfloorxLvl,CfloorMedShk)
        q_temp = np.exp(-b_temp)
        CfloorMedLvl = (CfloorxLvl/MedPriceNow)*q_temp/(1.+q_temp)
        CfloorcLvl = CfloorxLvl/(1.+q_temp)
        aLvlNow[NeedHelp] = 0.
        cLvlNow[NeedHelp] = CfloorcLvl
        MedLvlNow[NeedHelp] = CfloorMedLvl
        OOPnow[NeedHelp] = 0.
        WelfareNow = np.zeros_like(mLvlNow)
        WelfareNow[NeedHelp] = Welfare

        # Calculate end of period assets and store results as attributes of self
        self.mLvlNow = mLvlNow
        self.aLvlNow = aLvlNow
        self.cLvlNow = cLvlNow
        self.MedLvlNow = MedLvlNow
        self.OOPnow = OOPnow
        self.PremNow = PremNow
        self.ContractNow = ContractNow
        self.WelfareNow = WelfareNow
        
#    def calcInsurancePayments(self):
#        '''
#        Uses the results of a simulation to calculate the expected payments of
#        each insurance contract by age-health.  Requires that track_vars
#        includes 'MedLvlNow' and 'OOPnow' or will fail.  Results are stored
#        in the attributes ContractPayments and ContractCounts.
#        
#        Parameters
#        ----------
#        None
#        
#        Returns
#        -------
#        None
#        '''
#        # Get dimensions of output objects and initialize them
#        StateCount = self.MrkvArray[0].shape[0]
#        MaxContracts = max([max([len(self.ContractList[t][h]) for h in range(StateCount)]) for t in range(self.T_sim)])
#        ContractPayments = np.zeros((self.T_sim,StateCount,MaxContracts)) + np.nan
#        ContractCounts = np.zeros((self.T_sim,StateCount,MaxContracts),dtype=int)
#        
#        # Make arrays of payments by insurance and the indices of contracts and states
#        MedPrice_temp = np.tile(np.reshape(self.MedPrice[0:self.T_sim],(self.T_sim,1)),(1,self.AgentCount))
#        Payments = self.MedLvlNow_hist*MedPrice_temp - self.OOPnow_hist
#        Choices = self.ContractNow_hist
#        States = self.MrkvHist
#        Payments[States == -1] = 0.0 # Dead people have no costs
#        
#        # Calculate insurance payment and individual count for each age-state-contract
#        for j in range(StateCount):
#            temp = States == j
#            for z in range(MaxContracts):
#                these = np.logical_and(temp,Choices == z)
#                ContractPayments[:,j,z] = np.sum(Payments*these,axis=1)
#                ContractCounts[:,j,z] = np.sum(these,axis=1)
#                
#        # Store the results as attributes of self
#        self.ContractPayments = ContractPayments
#        self.ContractCounts = ContractCounts

        
    def calcExpInsPayByContract(self):
        '''
        Calculates the expected insurance payout for each contract in each health
        state at each age for this type.  Makes use of the premium functions
        in the attribute PremiumFuncs, which does not necessarily contain the
        premium functions used to solve the dynamic problem.  This function is
        called as part of marketAction to find "statically stable" premiums.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        None
        '''
        StateCount = self.MrkvArray[0].shape[0]
        MaxContracts = max([max([len(self.ContractList[t][h]) for h in range(StateCount)]) for t in range(self.T_sim)])
        ExpInsPay = np.zeros((self.T_sim,StateCount,MaxContracts))
        ExpBuyers = np.zeros((self.T_sim,StateCount,MaxContracts))
        
        for t in range(self.T_sim):
            random_choice = self.ChoiceShkMag[t] > 0.
            for j in range(StateCount):
                these = self.MrkvHist[t,:] == j
                N = np.sum(these)
                mLvl = self.mLvlNow_hist[t,these]
                pLvl = self.pLvlHist[t,these]
                Z = len(self.ContractList[t][j])
                vNvrs_array = np.zeros((N,Z))
                AV_array = np.zeros((N,Z))
                for z in range(Z):
                    Premium = np.maximum(self.PremiumFuncs[t][j][z](mLvl,pLvl) - self.PremiumSubsidy, 0.0) # net premium
                    mLvl_temp = mLvl - Premium
                    Unaffordable = mLvl_temp < 0.
                    AV_array[:,z] = self.AVfunc[t][j][z](mLvl_temp,pLvl)
                    vNvrs_array[:,z] = self.vFuncByContract[t][j][z].func(mLvl_temp,pLvl)
                    vNvrs_array[Unaffordable,z] = -np.inf
                    AV_array[Unaffordable,z] = 0.0
                if random_choice:
                    v_best = np.max(vNvrs_array,axis=1)
                    v_best_big = np.tile(np.reshape(v_best,(N,1)),(1,Z))
                    v_adj_exp = np.exp((vNvrs_array - v_best_big)/self.ChoiceShkMag[t])
                    v_sum_rep = np.tile(np.reshape(np.sum(v_adj_exp,axis=1),(N,1)),(1,Z))
                    ChoicePrbs = v_adj_exp/v_sum_rep
                else:
                    z_choice = np.argmax(vNvrs_array,axis=1)
                    ChoicePrbs = np.zeros((N,Z))
                    for z in range(Z): # is there a non-loop way to do this? YES THERE IS
                        ChoicePrbs[z_choice==z,z] = 1.0
                ExpInsPay[t,j,0:Z] = np.sum(ChoicePrbs*AV_array,axis=0)
                ExpBuyers[t,j,0:Z] = np.sum(ChoicePrbs,axis=0)
                
        self.ExpInsPay = ExpInsPay
        self.ExpBuyers = ExpBuyers
    
    
    def plotvFunc(self,t,p,mMin=0.0,mMax=10.0,H=None,decurve=True):
        mLvl = np.linspace(mMin,mMax,200)
        if H is None:
            H = range(len(self.LivPrb[0]))
        for h in H:
            if decurve:
                f = lambda x : self.solution[t].vFunc[h].func(x,p*np.ones_like(x))
            else:
                f = lambda x : self.solution[t].vFunc[h](x,p*np.ones_like(x))
            plt.plot(mLvl,f(mLvl))
        plt.xlabel('Market resources mLvl')
        if decurve:
            plt.ylabel('Pseudo-inverse value uinv(v(mLvl))')
        else:
            plt.ylabel('Value v(mLvl)')
        plt.show()
        
    def plotvPfunc(self,t,p,mMin=0.0,mMax=10.0,H=None,decurve=True):
        mLvl = np.linspace(mMin,mMax,200)
        if H is None:
            H = range(len(self.LivPrb[0]))
        for h in H:
            if decurve:
                f = lambda x : self.solution[t].vPfunc[h].UpperFunc.cFunc(x,p*np.ones_like(x))
            else:
                f = lambda x : self.solution[t].vPfunc[h](x,p*np.ones_like(x))
            plt.plot(mLvl,f(mLvl))
        plt.xlabel('Market resources mLvl')
        if decurve:
            plt.ylabel('Pseudo-inverse marg value uPinv(vP(mLvl))')
        else:
            plt.ylabel('Marginal value vP(mLvl)')
        plt.show()
        
    def plotvFuncByContract(self,t,h,p,mMin=0.0,mMax=10.0,Z=None):
        print('Pseudo-inverse value function by contract:')
        mLvl = np.linspace(mMin,mMax,200)
        if Z is None:
            Z = range(len(self.solution[t].vFuncByContract[h]))
        for z in Z:
            f = lambda x : self.solution[t].vFuncByContract[h][z].func(x,p*np.ones_like(x))
            Prem = self.ContractList[t][h][z].Premium(0)
            plt.plot(mLvl+Prem,f(mLvl))
        f = lambda x : self.solution[t].vFunc[h].func(x,p*np.ones_like(x))
        plt.plot(mLvl,f(mLvl),'-k')
        plt.xlabel('Market resources mLvl')
        plt.ylabel('Contract pseudo-inverse value uinv(v(mLvl))')
        plt.ylim(ymin=0.0)
        plt.show()
        
    def plotcFuncByContract(self,t,h,p,MedShk,mMin=0.0,mMax=10.0,Z=None):
        mLvl = np.linspace(mMin,mMax,200)
        if Z is None:
            Z = range(len(self.solution[t].policyFunc[h]))
        for z in Z:
            cLvl,MedLvl = self.solution[t].policyFunc[h][z](mLvl,p*np.ones_like(mLvl),MedShk*np.ones_like(mLvl))
            #plt.plot(mLvl[:-1],(cLvl[1:]-cLvl[0:-1])/(mLvl[1]-mLvl[0]))
            plt.plot(mLvl,cLvl)
        plt.xlabel('Market resources mLvl')
        plt.ylabel('Consumption c(mLvl)')
        plt.ylim(ymin=0.0)
        plt.show()
        
    def plotMedFuncByContract(self,t,h,p,MedShk,mMin=0.0,mMax=10.0,Z=None):
        mLvl = np.linspace(mMin,mMax,200)
        if Z is None:
            Z = range(len(self.solution[t].policyFunc[h]))
        for z in Z:
            cLvl,MedLvl = self.solution[t].policyFunc[h][z](mLvl,p*np.ones_like(mLvl),MedShk*np.ones_like(mLvl))
            #plt.plot(mLvl[:-1],(MedLvl[1:]-MedLvl[0:-1])/(mLvl[1]-mLvl[0]))
            plt.plot(mLvl,MedLvl)
        plt.xlabel('Market resources mLvl')
        plt.ylabel('Medical care Med(mLvl)')
        plt.ylim(ymin=0.0)
        plt.show()
        
    def plotcFuncByMedShk(self,t,h,z,p,mMin=0.0,mMax=10.0,MedShkSet=None):        
        mLvl = np.linspace(mMin,mMax,200)
        if MedShkSet is None:
            MedShkSet = self.MedShkDstn[t][h][1]
        for MedShk in MedShkSet:            
            cLvl,MedLvl = self.solution[t].policyFunc[h][z](mLvl,p*np.ones_like(mLvl),MedShk*np.ones_like(mLvl))
            plt.plot(mLvl,cLvl)
        plt.xlabel('Market resources mLvl')
        plt.ylabel('Consumption c(mLvl)')
        plt.ylim(ymin=0.0)
        plt.show()
        
    def plotMedFuncByMedShk(self,t,h,z,p,mMin=0.0,mMax=10.0,MedShkSet=None):        
        mLvl = np.linspace(mMin,mMax,200)
        if MedShkSet is None:
            MedShkSet = self.MedShkDstn[t][h][1]
        for MedShk in MedShkSet:            
            cLvl,MedLvl = self.solution[t].policyFunc[h][z](mLvl,p*np.ones_like(mLvl),MedShk*np.ones_like(mLvl))
            plt.plot(mLvl,MedLvl)
        plt.xlabel('Market resources mLvl')
        plt.ylabel('Medical care Med(mLvl)')
        plt.ylim(ymin=0.0)
        plt.show()
        
    def plotxFuncByMedShk(self,t,h,z,p,mMin=0.0,mMax=10.0,MedShkSet=None):        
        mLvl = np.linspace(mMin,mMax,200)
        if MedShkSet is None:
            MedShkSet = self.MedShkDstn[t][h][1]
        for MedShk in MedShkSet:            
            xLvl = self.solution[t].policyFunc[h][z].xFunc(mLvl,p*np.ones_like(mLvl),MedShk*np.ones_like(mLvl))
            plt.plot(mLvl,xLvl)
        plt.xlabel('Market resources mLvl')
        plt.ylabel('Expenditure x(mLvl)')
        plt.ylim(ymin=0.0)
        plt.show()
        
    def plotcEffFuncByMedShk(self,t,h,z,p,mMin=0.0,mMax=10.0,MedShkSet=None):        
        mLvl = np.linspace(mMin,mMax,200)
        if MedShkSet is None:
            MedShkSet = self.MedShkDstn[t][h][1]
        for MedShk in MedShkSet:            
            cLvl,MedLvl = self.solution[t].policyFunc[h][z](mLvl,p*np.ones_like(mLvl),MedShk*np.ones_like(mLvl))
            cEffLvl = (1. - np.exp(-(MedLvl+1e-10)/MedShk))**self.CRRAmed*cLvl
            plt.plot(mLvl,cEffLvl)
        plt.xlabel('Market resources mLvl')
        plt.ylabel('Effective consumption C(mLvl)')
        plt.ylim(ymin=0.0)
        plt.show()
        
    def plotAVfuncByContract(self,t,h,p,mMin=0.0,mMax=10.0,Z=None):
        print('Actuarial value function by contract:')
        mLvl = np.linspace(mMin,mMax,200)
        if Z is None:
            Z = range(len(self.solution[t].AVfunc[h]))
        for z in Z:
            f = lambda x : self.solution[t].AVfunc[h][z].func(x,p*np.ones_like(x))
            plt.plot(mLvl,f(mLvl))
        plt.xlabel('Market resources mLvl')
        plt.ylabel('Contract actuarial value')
        plt.ylim(ymin=0.0)
        plt.show()
        
    
class InsSelStaticConsumerType(InsSelConsumerType):
    '''
    A class to represent consumer types in the "static" version of the insurance
    selection model.  Extends / simplifies the standard InsSelConsumerType.
    '''
    def __init__(self,cycles=1,time_flow=True,**kwds):
        '''
        Instantiate a new InsSelConsumerType with given data, and construct objects
        to be used during solution (income distribution, assets grid, etc).
        See ConsumerParameters.init_med_shock for a dictionary of the keywords
        that should be passed to the constructor.
        
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
        # Initialize a basic AgentType
        InsSelConsumerType.__init__(self,**kwds)
        self.solveOnePeriod = solveInsuranceSelectionStatic # Choose correct solver
        self.PermIncCount *= 4
        self.PermInc_tail_N *= 3
           
    def update(self):
        '''
        Make constructed inputs for solving the model.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        None
        '''
        self.updateAssetsGrid()
        self.updatePermIncGrid()
        self.updateMedShockProcess()
        self.updateIncomeProcess()
        self.makeMasterbFromxFuncs()
        self.makeMasterDfuncs()
        self.distributeConstructedFuncs()
        self.makevFuncsAndPolicyFuncsByCopay()
        self.makePolicyFuncList()
        
    def updateSolutionTerminal(self):
        return None
        
    def updatePermIncGrid(self):
        InsSelConsumerType.updatePermIncGrid(self)
        self.xLvlGrid = deepcopy(self.pLvlGrid)
        self.addToTimeVary('xLvlGrid')
        
    def makevFuncsAndPolicyFuncsByCopay(self):
        '''
        Constructs value and policy functions for each copayment rate (really
        effective medical price) in the agent's lifecycle.  Can only be run
        after makeMasterbFromxFuncs().  Stores results in PolicyFuncByCopay and
        vFuncByCopay.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        None
        '''
        SpendAllFunc = IdentityFunction(n_dims=3)
        PolicyFuncByCopay = []
        vFuncByCopay = []
        
        for k in range(len(self.CopayListAll)):
            # Construct 2D grids for this effective price of care
            MedPriceEff = self.CopayListAll[k]
            xLvlVec = self.bFromxFuncListAll[k].x_list
            ShkVec = self.bFromxFuncListAll[k].y_list
            xLvlGrid = np.tile(np.reshape(xLvlVec,(xLvlVec.size,1)),(1,ShkVec.size))
            ShkGrid = np.tile(np.reshape(ShkVec,(1,ShkVec.size)),(xLvlVec.size,1))
            
            # Find consumption and medical care at each gridpoint
            bGrid = self.bFromxFuncListAll[k].f_values
            qGrid = np.exp(-bGrid)
            cLvlGrid = xLvlGrid/(1.+qGrid)
            MedLvlGrid = (xLvlGrid/MedPriceEff)*qGrid/(1.+qGrid)
            
            # Calculate the value of each gridpoint and take pseudo inverse
            vNvrsGrid = (1.-np.exp(-MedLvlGrid/ShkGrid))**self.CRRAmed*cLvlGrid
            below_C_floor = vNvrsGrid < self.Cfloor
            vNvrsGrid[below_C_floor] = self.Cfloor
            ShkZero = ShkGrid == 0.0
            vNvrsGrid[ShkZero] = cLvlGrid[ShkZero]
            vNvrsGrid[0,:] = 0.0
            vNvrsGridX = np.tile(np.reshape(vNvrsGrid,(xLvlVec.size,1,ShkVec.size)),(1,2,1))
            vNvrsFunc = TrilinearInterp(vNvrsGridX,xLvlVec,np.array([0.0,100.0]),ShkVec)
            
            # Make policy and value functions
            policyFuncThisCopay = MedShockPolicyFuncPrecalc(SpendAllFunc,self.bFromxFuncListAll[k],MedPriceEff)
            PolicyFuncByCopay.append(policyFuncThisCopay)
            vFuncThisCopay = ValueFunc3D(vNvrsFunc,self.CRRA)
            vFuncByCopay.append(vFuncThisCopay)
        
        # Store results in self
        self.PolicyFuncByCopay = PolicyFuncByCopay
        self.vFuncByCopay = vFuncByCopay
        
        
    def makePolicyFuncList(self):
        '''
        Construct InsSelPolicyFuncs for each contract and distribute them into
        lifecycle lists (by health state).  Stores the result in PolicyFuncList
        
        Parameters
        ----------
        None
        
        Returns
        -------
        None
        '''
        PolicyFuncList = []
        for t in range(self.T_cycle):
            policyFuncs_this_t = []
            for j in range(len(self.ContractList[t])):
                policyFuncs_this_state = []
                for Contract in self.ContractList[t][j]:
                    FullPrice = Contract.MedPrice
                    CopayPrice = Contract.Copay*Contract.MedPrice
                    idx0 = np.argwhere(np.array(self.CopayListAll)==FullPrice)[0][0]
                    idx1 = np.argwhere(np.array(self.CopayListAll)==CopayPrice)[0][0]
                    policyFunc = InsSelPolicyFunc(ValueFuncFullPrice=self.vFuncByCopay[idx0],
                                                  ValueFuncCopay=self.vFuncByCopay[idx1],
                                                  PolicyFuncFullPrice=self.PolicyFuncByCopay[idx0],
                                                  PolicyFuncCopay=self.PolicyFuncByCopay[idx1],
                                                  DpFuncFullPrice=self.DpFuncListAll[idx0],
                                                  DpFuncCopay=self.DpFuncListAll[idx1],
                                                  Contract=Contract,
                                                  CRRAmed=self.CRRAmed)
                    policyFuncs_this_state.append(policyFunc)
                policyFuncs_this_t.append(policyFuncs_this_state)
            PolicyFuncList.append(policyFuncs_this_t)
        self.PolicyFuncList = PolicyFuncList
        self.addToTimeVary('PolicyFuncList')

                         
    def simOnePeriod(self):
        '''
        Simulates one period of the static insurance selection model.  The
        simulator uses the "history" style of simulation and should only be run
        after executing both the solve() and makeShockHistory() methods.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        None
        '''
        t = self.t_sim
        
        # Get state and shock vectors for (living) agents
        yLvlNow = self.pLvlHist[t,:]
        HealthNow = self.MrkvHist[t,:]
        MedShkNow = self.MedShkHist[t,:]
        PrefShkNow = self.PrefShkHist[t,:]

        # Loop through each health state and get agents' controls
        cLvlNow = np.zeros_like(yLvlNow) + np.nan
        OOPnow = np.zeros_like(yLvlNow) + np.nan
        MedLvlNow = np.zeros_like(yLvlNow) + np.nan
        xLvlNow = np.zeros_like(yLvlNow) + np.nan
        PremNow = np.zeros_like(yLvlNow) + np.nan
        ContractNow = -np.ones(self.AgentCount,dtype=int)
        for h in range(5):
            these = HealthNow == h
            N = np.sum(these)
            Z = len(self.solution[t].policyFunc[h])
            
            # Get the pseudo-inverse value of holding each contract
            vNvrs_temp = np.zeros((N,Z)) + np.nan
            m_temp = yLvlNow[these]
            p_temp = yLvlNow[these]
            for z in range(Z):                
                Premium = np.maximum(self.PremiumFuncs[t][h][z](m_temp)  - self.PremiumSubsidy,0.0)
                x_temp = m_temp - Premium
                if self.DecurveBool:
                    vNvrs_temp[:,z] = self.solution[t].vFuncByContract[h][z].func(x_temp)
                else:
                    vNvrs_temp[:,z] = self.solution[t].vFuncByContract[h][z](x_temp)
                vNvrs_temp[x_temp<0.,z] = -np.inf
            
            # Get choice probabilities for each contract
            random_choice = self.ChoiceShkMag[t] > 0.
            vNvrs_temp[np.isnan(vNvrs_temp)] = -np.inf
            if random_choice:
                v_best = np.max(vNvrs_temp,axis=1)
                v_best_big = np.tile(np.reshape(v_best,(N,1)),(1,Z))
                v_adj_exp = np.exp((vNvrs_temp - v_best_big)/self.ChoiceShkMag[0]) # THIS NEEDS TO LOOK UP t_cycle TO BE CORRECT IN ALL CASES
                v_sum_rep = np.tile(np.reshape(np.sum(v_adj_exp,axis=1),(N,1)),(1,Z))
                ChoicePrbs = v_adj_exp/v_sum_rep
                Cutoffs = np.cumsum(ChoicePrbs,axis=1)
                
                # Select a contract for each agent based on the unified preference shock
                PrefShk_temp = np.tile(np.reshape(PrefShkNow[these],(N,1)),(1,Z))
                z_choice = np.sum(PrefShk_temp > Cutoffs,axis=1).astype(int)
            else:
                z_choice = np.argmax(vNvrs_temp,axis=1)
            
            # For each contract, get controls for agents who buy it
            c_temp = np.zeros(N) + np.nan
            Med_temp = np.zeros(N) + np.nan
            OOP_temp = np.zeros(N) + np.nan
            Prem_temp = np.zeros(N) + np.nan
            MedShk_temp = MedShkNow[these]
            for z in range(Z):
                idx = z_choice == z
                Prem_temp[idx] = np.maximum(self.PremiumFuncs[t][h][z](m_temp[idx],p_temp[idx]) - self.PremiumSubsidy, 0.0)
                c_temp[idx],Med_temp[idx] = self.solution[t].policyFunc[h][z](m_temp[idx]-Prem_temp[idx],p_temp[idx],MedShk_temp[idx])
                Med_temp[idx] = np.maximum(Med_temp[idx],0.0) # Prevents numeric glitching
                OOP_temp[idx] = self.solution[t].policyFunc[h][z].Contract.OOPfunc(Med_temp[idx])            
            
            # Store the controls for this health
            cLvlNow[these] = c_temp
            MedLvlNow[these] = Med_temp
            xLvlNow[these] = c_temp + OOP_temp
            OOPnow[these] = OOP_temp
            PremNow[these] = Prem_temp
            ContractNow[these] = z_choice

        # Calculate end of period assets and store results as attributes of self
        self.aLvlNow = np.zeros_like(cLvlNow)
        self.cLvlNow = cLvlNow
        self.MedLvlNow = MedLvlNow
        self.OOPnow = OOPnow
        self.PremNow = PremNow
        self.ContractNow = ContractNow

        
    def calcExpInsPaybyContract(self):
        '''
        Calculates the expected insurance payout for each contract in each health
        state at each age for this type.  Makes use of the premium functions
        in the attribute PremiumFuncs, which does not necessarily contain the
        premium functions used to solve the dynamic problem.  This function is
        called as part of marketAction to find "statically stable" premiums.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        None
        '''
        StateCount = self.MrkvArray[0].shape[0]
        MaxContracts = max([max([len(self.ContractList[t][h]) for h in range(StateCount)]) for t in range(self.T_sim)])
        ExpInsPay = np.zeros((self.T_sim,StateCount,MaxContracts))
        ExpBuyers = np.zeros((self.T_sim,StateCount,MaxContracts))
        
        for t in range(self.T_sim):
            random_choice = self.ChoiceShkMag[0] > 0.
            for j in range(StateCount):
                these = self.HealthBoolArray[t,:,j]
                N = np.sum(these)
                pLvl = self.pLvlHist[t,these]
                Z = len(self.solution[t].AVfunc[j])
                vNvrs_array = np.zeros((N,Z))
                AV_array = np.zeros((N,Z))
                for z in range(Z):
                    Premium = np.maximum(self.PremiumFuncs[t][j][z](pLvl) - self.PremiumSubsidy, 0.0)
                    xLvl_temp = pLvl - Premium
                    AV_array[:,z] = self.solution[t].AVfunc[j][z](xLvl_temp)
                    if self.DecurveBool:
                        vNvrs_array[:,z] = self.solution[t].vFuncByContract[j][z].func(xLvl_temp)
                    else:
                        vNvrs_array[:,z] = self.solution[t].vFuncByContract[j][z](xLvl_temp)
                    unaffordable = xLvl_temp < 0.
                    AV_array[unaffordable,z] = 0.0
                    vNvrs_array[unaffordable,z] = -np.inf
                if random_choice:
                    v_best = np.max(vNvrs_array,axis=1)
                    v_best_big = np.tile(np.reshape(v_best,(N,1)),(1,Z))
                    v_adj_exp = np.exp((vNvrs_array - v_best_big)/self.ChoiceShkMag[0]) # THIS NEEDS TO LOOK UP t_cycle TO BE CORRECT IN ALL CASES
                    v_sum_rep = np.tile(np.reshape(np.sum(v_adj_exp,axis=1),(N,1)),(1,Z))
                    ChoicePrbs = v_adj_exp/v_sum_rep
                else:
                    z_choice = np.argmax(vNvrs_array,axis=1)
                    ChoicePrbs = np.zeros((N,Z))
                    for z in range(Z): # is there a non-loop way to do this?
                        ChoicePrbs[z_choice==z,z] = 1.0
                ExpInsPay[t,j,0:Z] = np.sum(ChoicePrbs*AV_array,axis=0)
                ExpBuyers[t,j,0:Z] = np.sum(ChoicePrbs,axis=0)
                
        self.ExpInsPay = ExpInsPay
        self.ExpBuyers = ExpBuyers
        
        
    def makeStynamicValueFunc(self):
        '''
        Constructs a "stynamic" value function over health and permanent income
        in each period.  Agents behave according to the quasi-static model, spending
        all income each period on premiums, consumption, and medical care, but
        calculate their expected lifetime utility dynamically, using transition
        probabilities on the permanent income process and health state.  Adds
        the attribute vFunc to each element of the attribute solution.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        None
        '''
        time_orig = self.time_flow
        self.timeRev()
        u = lambda x : CRRAutility(x,gam=self.CRRA)
        uinv = lambda x : CRRAutility_inv(x,gam=self.CRRA)
        
        # Initialize the stynamic value function; in the terminal period, we didn't
        # bother to solve the consumer's problem (no one lives that long).
        vFuncStynamicNew = None

        # Loop through each non-terminal period, constructing stynamic value function
        for t in range(0,self.T_cycle):
            vFuncStynamicPrev = vFuncStynamicNew
            StateCount = len(self.solution[t+1].vFuncByContract)
            pLvlGrid = self.xLvlGrid[t]
            pLvlCount = pLvlGrid.size
            
            # In the last non-terminal period, we impose that there is no future
            if vFuncStynamicPrev is None:
                EndOfPrdv = np.zeros((pLvlCount,StateCount))
                                            
            else: # Otherwise, we must construct the end-of-period value function
                # Calculate end-of-period value over permanent income *conditional*
                # on next period's discrete health state.
                vNextCond = np.zeros((pLvlCount,StateCount)) + np.nan
                for j in range(StateCount):
                    ShkPrbsNext      = self.IncomeDstn[t][j][0]
                    PermShkValsNext  = self.IncomeDstn[t][j][1]
                    ShkCount         = PermShkValsNext.size  
                    vFuncNext        = vFuncStynamicPrev[j]
                    
                    # Calculate expected value conditional on achieving this future health state
                    pLvlNext = np.tile(np.reshape(pLvlGrid,(pLvlCount,1)),(1,ShkCount))**self.PermIncCorr*np.tile(np.reshape(PermShkValsNext,(1,ShkCount)),(pLvlCount,1))
                    vNextCond[:,j] = np.sum(vFuncNext(pLvlNext)*ShkPrbsNext,axis=1)
                    
                # Use the discrete transition probabilities to calculate end-of-period value before health transition
                EndOfPrdv = np.dot(self.MrkvArray[t],vNextCond.transpose()).transpose()
                for j in range(StateCount):
                    DiscFacEff = self.DiscFac*self.LivPrb[t][j] # "effective" discount factor
                    EndOfPrdv[:,j] = DiscFacEff*EndOfPrdv[:,j]

            # Get current period expected utility / static value from purchasing each contract
            ContractCounts = np.zeros(StateCount,dtype=int)
            for j in range(StateCount):
                ContractCounts[j] = len(self.solution[t+1].vFuncByContract[j])
            ContractCount = np.max(ContractCounts)
            UnvrsNow = np.zeros((pLvlCount,StateCount,ContractCount))
            for j in range(StateCount):
                for z in range(ContractCounts[j]):
                    PremiumGrid = self.ContractList[t][j][z].Premium(pLvlGrid)
                    xLvlGrid = pLvlGrid - PremiumGrid
                    if self.DecurveBool:
                        UnvrsNow[:,j,z] = self.solution[t+1].vFuncByContract[j][z].func(xLvlGrid)
                    else:
                        UnvrsNow[:,j,z] = self.solution[t+1].vFuncByContract[j][z](xLvlGrid)
                    UnvrsNow[xLvlGrid < 0.0,j,z] = -np.inf
                    
            # Calculate expected value across contracts (from preference shocks)
            U_best = np.max(UnvrsNow,axis=2)
            U_exp = np.exp((UnvrsNow - np.tile(np.reshape(U_best,(pLvlCount,StateCount,1)),(1,1,ContractCount)))/self.ChoiceShkMag[t])
            U_exp_sum = np.nansum(U_exp,axis=2)
            UnvrsExp = np.log(U_exp_sum)*self.ChoiceShkMag[t] + U_best
            UtilityExp = u(UnvrsExp)
            
            # Combine current period "static value" with future "stynamic value" and construct value functions
            ValueArray = UtilityExp + EndOfPrdv
            vFuncStynamicNew = []
            for j in range(StateCount):
                vNvrs_temp = uinv(ValueArray[:,j])
                vNvrs_temp[0] = 0.0 # Fix issue at bottom
                vFuncNvrs_contract = LinearInterp(pLvlGrid,vNvrs_temp)
                vFuncStynamicNew.append(ValueFunc(vFuncNvrs_contract,self.CRRA))
                
            # Add the value function for the solution for this period
            self.solution[t+1].vFunc = vFuncStynamicNew

        # Restore the original flow of time
        if time_orig:
            self.timeFwd()
             
              
####################################################################################################
        
if __name__ == '__main__':
    import InsuranceSelectionParameters as Params
    mystr = lambda number : "{:.4f}".format(number)
            
    # Make an example type
    MyType = InsSelConsumerType(**Params.init_insurance_selection)
    
    # Make medical insurance contracts
    NullContract = MedInsuranceContract(ConstantFunction(0.0),0.0,1.0,MyType.MedPrice[0])
    ContractLoDeduct = MedInsuranceContract(ConstantFunction(0.1),0.5,0.1,MyType.MedPrice[0])
    ContractHiDeduct = MedInsuranceContract(ConstantFunction(0.05),2.0,0.1,MyType.MedPrice[0])
    ContractOther    = MedInsuranceContract(ConstantFunction(0.1),0.0,0.2,MyType.MedPrice[0])
    #Contracts = [NullContract,ContractLoDeduct,ContractHiDeduct,ContractOther]
    Contracts = [NullContract,ContractLoDeduct]
    #Contracts = [NullContract]
    MyType.ContractList = MyType.T_cycle*[5*[Contracts]]
    
#    t_start = clock()
#    MyType.update()
#    t_end = clock()
#    print('Updating the agent took ' + mystr(t_end-t_start) + ' seconds.')
    
    t_start = clock()
    MyType.solve()
    t_end = clock()
    print('Solving the agent took ' + mystr(t_end-t_start) + ' seconds.')
    
#    print('Pseudo-inverse value function by contract:')
#    mLvl = np.linspace(0,5,200)
#    h = 1
#    J = len(MyType.solution_terminal.vFuncByContract[0])
#    for j in range(J):
#        f = lambda x : MyType.solution_terminal.vFuncByContract[h][j].func(x,np.ones_like(x))
#        plt.plot(mLvl+Contracts[j].Premium(1,1,1),f(mLvl))
#    f = lambda x : MyType.solution_terminal.vFunc[h].func(x,np.ones_like(x))
#    plt.plot(mLvl,f(mLvl))
#    plt.show()    
#    
#    print('Marginal pseudo-inverse value function by contract:')
#    mLvl = np.linspace(0,.25,200)
#    J = len(MyType.solution_terminal.vFuncByContract[0])
#    for j in range(J):
#        f = lambda x : MyType.solution_terminal.vFuncByContract[0][j].func.derivativeX(x,np.ones_like(x))
#        plt.plot(mLvl,f(mLvl))
#    plt.show()
#    
#    print('Pseudo-inverse value function by health:')
#    mLvl = np.linspace(0,5,200)
#    H = len(MyType.LivPrb[0])
#    for h in range(H):
#        f = lambda x : MyType.solution_terminal.vFunc[h].func(x,np.ones_like(x))
#        plt.plot(mLvl,f(mLvl))
#    plt.show()
#    
#    print('Marginal pseudo-inverse value function by health:')
#    mLvl = np.linspace(0,0.25,200)
#    J = len(MyType.LivPrb[0])
#    for j in range(J):
#        f = lambda x : MyType.solution_terminal.vFunc[0].func.derivativeX(x,np.ones_like(x))
#        plt.plot(mLvl,f(mLvl))
#    plt.show()
#    
#    print('Pseudo-inverse marginal value function by health:')
#    mLvl = np.linspace(0,0.25,200)
#    J = len(MyType.LivPrb[0])
#    for j in range(J):
#        f = lambda x : MyType.solution_terminal.vPfunc[0].cFunc(x,np.ones_like(x))
#        plt.plot(mLvl,f(mLvl))
#    plt.show()
#    
#    print('Marginal pseudo-inverse marginal value function by health:')
#    mLvl = np.linspace(0,0.25,200)
#    J = len(MyType.LivPrb[0])
#    for j in range(J):
#        f = lambda x : MyType.solution_terminal.vPfunc[0].cFunc.derivativeX(x,np.ones_like(x))
#        plt.plot(mLvl,f(mLvl))
#    plt.show()
#    
#    print('Consumption function by contract:')
#    mLvl = np.linspace(0,5,200)
#    J = len(MyType.solution_terminal.vFuncByContract[0])
#    for j in range(J):
#        cLvl,MedLvl = MyType.solution_terminal.policyFunc[0][j](mLvl,np.ones_like(mLvl),1*np.ones_like(mLvl))
#        plt.plot(mLvl,cLvl)
#    plt.show()
#    
#    print('Pseudo-inverse value function by coinsurance rate:')
#    mLvl = np.linspace(0,5,200)
#    J = len(MyType.solution_terminal.vFuncByContract[0])
#    for j in range(J):
#        v = MyType.solution_terminal.policyFunc[0][j].ValueFuncCopay.vFuncNvrs(mLvl,np.ones_like(mLvl),1*np.ones_like(mLvl))
#        plt.plot(mLvl,v)
#    plt.show()
#    
#    print('Marginal pseudo-inverse value function by coinsurance rate:')
#    mLvl = np.linspace(0,5,200)
#    J = len(MyType.solution_terminal.vFuncByContract[0])
#    for j in range(J):
#        v = MyType.solution_terminal.policyFunc[0][j].ValueFuncCopay.vFuncNvrs.derivativeX(mLvl,np.ones_like(mLvl),1*np.ones_like(mLvl))
#        plt.plot(mLvl,v)
#    plt.show()
    
    p = 1.0    
    
    print('Pseudo-inverse value function by health:')
    mLvl = np.linspace(0.0,10,200)
    H = len(MyType.LivPrb[0])
    for h in range(H):
        f = lambda x : MyType.solution[0].vFunc[h].func(x,p*np.ones_like(x))
        plt.plot(mLvl,f(mLvl))
    plt.show()
    
    print('Pseudo-inverse marginal value function by health:')
    mLvl = np.linspace(0,10,200)
    H = len(MyType.LivPrb[0])
    for h in range(H):
        f = lambda x : MyType.solution[0].vPfunc[h].cFunc(x,p*np.ones_like(x))
        plt.plot(mLvl,f(mLvl))
    plt.show()
    
    h = 0    
    
    print('Pseudo-inverse value function by contract:')
    mLvl = np.linspace(0,10,200)
    J = len(MyType.solution[0].vFuncByContract[h])
    for j in range(J):
        f = lambda x : MyType.solution[0].vFuncByContract[h][j].func(x,p*np.ones_like(x))
        plt.plot(mLvl+Contracts[j].Premium(1,1,1),f(mLvl))
    f = lambda x : MyType.solution[0].vFunc[h].func(x,p*np.ones_like(x))
    plt.plot(mLvl,f(mLvl),'-k')
    plt.show()
    
    MedShk = 1e-1    
    
    print('Pseudo-inverse value function by coinsurance rate:')
    mLvl = np.linspace(0,10,200)
    J = len(MyType.solution[0].vFuncByContract[h])
    for j in range(J):
        v = MyType.solution[0].policyFunc[h][j].ValueFuncCopay.vFuncNvrs(mLvl,p*np.ones_like(mLvl),MedShk*np.ones_like(mLvl))
        plt.plot(mLvl,v)
    plt.show()
    
    print('Consumption function by coinsurance rate:')
    mLvl = np.linspace(0,2,200)
    J = len(MyType.solution[0].vFuncByContract[h])
    for j in range(J):
        cLvl,MedLvl = MyType.solution[0].policyFunc[h][j].PolicyFuncCopay(mLvl,p*np.ones_like(mLvl),MedShk*np.ones_like(mLvl))
        plt.plot(mLvl,cLvl)
    plt.show()
    
    print('Consumption function by contract:')
    mLvl = np.linspace(0,10,200)
    J = len(MyType.solution[0].vFuncByContract[0])
    for j in range(J):
        cLvl,MedLvl = MyType.solution[0].policyFunc[h][j](mLvl,p*np.ones_like(mLvl),MedShk*np.ones_like(mLvl))
        plt.plot(mLvl,cLvl)
    plt.show()
