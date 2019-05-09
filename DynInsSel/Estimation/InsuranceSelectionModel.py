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
from HARKutilities import CRRAutility, CRRAutilityP, CRRAutilityPP, CRRAutilityP_inv, combineIndepMrkvArrays,\
                          CRRAutility_invP, CRRAutility_inv, CRRAutilityP_invP, NullFunc, makeGridExpMult
from HARKinterpolation import LinearInterp, CubicInterp, LinearInterpOnInterp1D, LinearInterpOnInterp2D,\
                              UpperEnvelope, ConstantFunction, CompositeFunc2D, \
                              VariableLowerBoundFunc2D, VariableLowerBoundFunc3D, VariableLowerBoundFunc3Dalt, \
                              BilinearInterp, CompositeFunc3D, IdentityFunction, LowerEnvelope2D, UpperEnvelope2D
from HARKsimulation import drawUniform, drawNormal, drawMeanOneLognormal, drawDiscrete, drawBernoulli
from ConsMedModel import MedShockConsumerType
from ConsIndShockModel import ValueFunc
from ConsPersistentShockModel import ValueFunc2D, MargValueFunc2D
from ConsMarkovModel import MarkovConsumerType
from ConsIndShockModel import constructLognormalIncomeProcessUnemployment
from JorgensenDruedahl import makeGridDenser, JDfixer, JDfixerSimple
from ValueFuncCL import ValueFuncCL
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.special import erfc
                                     
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
    
    
class TransConShareFunc(HARKobject):
    '''
    A class for representing the "transformed consumption share" function with
    "double CRRA" utility.  Instances of this class take as inputs xLvl and MedShkEff
    and return a transformed consumption share b.  By definition, optimal consumption
    cLvl = xLvl/(1 + exp(-b)).  MedShkEff = MedShk*EffPrice = MedShk*Price*Copay.
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


class RescaledFunc2D(HARKobject):
    '''
    A class for representing 2D functions f(x,y) with the properties lim_{x --> -inf} f(x,y) = g(y)
    and lim_{x --> inf} f(x,y) = 0.
    '''
    def __init__(self,ScaledFunc,LimitFunc):
        self.ScaledFunc = ScaledFunc
        self.LimitFunc  = LimitFunc
        
    def __call__(self,x,y):
        temp = self.ScaledFunc(x,y)
        return self.LimitFunc(y)/(1. + np.exp(temp))


class InsuranceSelectionSolution(HARKobject):
    '''
    Class for representing the single period solution of the insurance selection model.
    '''
    distance_criteria = ['vPfunc']
    
    def __init__(self, policyFunc=None, vFunc=None, vFuncByContract=None, vPfunc=None, vPPfunc=None,
                 AVfunc=None, CritDevFunc=None, mLvlMin=None, hLvl=None):
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
        AVfunc : [[HARKinterpolator2D]]
            Actuarial value as a function of permanent income and market resources, by health and contract.
        CritDevFunc : [[HARKinterpolator2D]]
            Critical deviation (from mean log medical needs shock) where consumption floor begins
            to bind, as a function of market resources and permanent income; list is by health state and contract.
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
        if CritDevFunc is None:
            CritDevFunc = []
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
        self.CritDevFunc  = copy(CritDevFunc)
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
        
        
    def appendSolution(self,policyFunc=None, vFunc=None, vFuncByContract=None, vPfunc=None, vPPfunc=None, AVfunc=None, CritDevFunc=None, hLvl=None):
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
        CritDevFunc : [HARKinterpolator2D]
            Critical deviation (from mean log medical needs shock) where consumption floor begins
            to bind, as a function of market resources and permanent income, by contract.
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
        self.CritDevFunc.append(CritDevFunc)
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
    
    def __init__(self,ValueFuncZeroShk,ValueFuncFullPrice,ValueFuncCopay,
                 cFuncZeroShk,PolicyFuncFullPrice,PolicyFuncCopay,
                 Contract,CRRA):
        '''
        Make a new instance of ValueFuncContract.
        
        Parameters
        ----------
        ValueFuncZeroShk : ValueFunc2D
            Value function (over market resources after premiums and permanent income) when medical
            needs shock is zero (Dev = -np.inf).
        ValueFuncFullPrice : ValueFunc3D
            Value function (over market resources after premiums / "option cost", permanent income,
            and medical shock) when the individual pays full price for care.
        ValueFuncCopay : ValueFunc3D
            Value function (over market resources after premiums / "option cost", permanent income,
            and medical shock) when the individual pays the coinsurance rate for care.
        cFuncZeroShk : HARKinterpolator2D
            Consumption function (over market resources after premiums and permanent income) when
            medical needs shock is zero (Dev = -np.inf).
        PolicyFuncFullPrice : cAndMedFunc
            Policy function when paying full price for care, including consumption and medical care,
            defined over market resources (after option cost), permanent income, and medical shock.
        PolicyFuncFullPrice : cAndMedFunc
            Policy function when paying the coinsurance rate for care, including consumption and
            medical care, defined over market resources, permanent income, and medical shock.
        Contract : MedInsuranceContract
            Medical insurance contract of interest.
        CRRA : float
            Coefficient of relative risk aversion.
            
        Returns
        -------
        None
        '''
        self.ValueFuncZeroShk = ValueFuncZeroShk
        self.ValueFuncFullPrice = ValueFuncFullPrice
        self.ValueFuncCopay = ValueFuncCopay
        self.cFuncZeroShk = cFuncZeroShk
        self.PolicyFuncFullPrice = PolicyFuncFullPrice
        self.PolicyFuncCopay = PolicyFuncCopay
        self.Contract = Contract
        self.CRRA = CRRA
        self.OptionCost = Contract.Deductible*(1.0-Contract.Copay)
               
    def __call__(self,mLvl,pLvl,Dev):
        '''
        Evaluate the policy function for this contract.
        
        Parameters
        ----------
        mLvl : np.array
             Market resource levels.
        pLvl : np.array
             Permanent income levels.
        Dev : np.array
             Standard deviations from mean log medical need shocks; can be -np.inf.
             
        Returns
        -------
        cLvl : np.array
            Optimal consumption at each point in the input.
        MedLvl : np.array
            Optimal medical care at each point in the input.
        xLvl : np.array
            Optimal total expenditure at each point in the input.
        '''
        if mLvl.shape is ():
            mLvl = np.array([mLvl])
            pLvl = np.array([pLvl])
            Dev = np.array([Dev])
            float_in = True
        else:
            float_in = False
            
        # Initialize output
        cLvl = np.zeros_like(mLvl)
        MedLvl = np.zeros_like(mLvl)
        xLvl = np.zeros_like(mLvl)
            
        # Determine which inputs have zero medical shock and fill in output
        ZeroShk = np.logical_and(np.isinf(Dev), Dev < 0.)
        cLvl[ZeroShk] = self.cFuncZeroShk(mLvl[ZeroShk],pLvl[ZeroShk])
        xLvl[ZeroShk] = cLvl[ZeroShk]
        
        # Get value of paying full price or paying "option cost" to pay coinsurance rate
        these = np.logical_not(ZeroShk)
        mTemp = mLvl[these]
        mTempAlt = mTemp-self.OptionCost
        pTemp = pLvl[these]
        DevTemp = Dev[these]
        v_Copay = self.ValueFuncCopay(mTempAlt,pTemp,DevTemp)
        v_Copay[mTempAlt < 0.] = -np.inf
        if self.OptionCost > 0.:
            v_FullPrice = self.ValueFuncFullPrice(mTemp,pTemp,DevTemp)
        
        # Decide which option is better and initialize output
        if self.OptionCost > 0.:
            Copay_better = v_Copay > v_FullPrice
        else:
            Copay_better = np.ones_like(v_Copay,dtype=bool)
        FullPrice_better = np.logical_not(Copay_better)
        
        # Fill in output using better of two choices
        cTemp = np.zeros_like(mTemp)
        MedTemp = np.zeros_like(mTemp)
        xTemp = np.zeros_like(mTemp)
        cTemp[Copay_better], MedTemp[Copay_better], xTemp[Copay_better] = self.PolicyFuncCopay(mTempAlt[Copay_better],pTemp[Copay_better],DevTemp[Copay_better])
        cTemp[FullPrice_better], MedTemp[FullPrice_better], xTemp[FullPrice_better] = self.PolicyFuncFullPrice(mTemp[FullPrice_better],pTemp[FullPrice_better],DevTemp[FullPrice_better])
        xTemp[Copay_better] += self.OptionCost
        #MedTemp[Copay_better] += self.OptionCost
        cLvl[these] = cTemp
        MedLvl[these] = MedTemp
        xLvl[these] = xTemp
        
        if float_in:
            return cLvl[0], MedLvl[0], xLvl[0]
        else:
            return cLvl, MedLvl, xLvl
        
        
    def cFunc(self,mLvl,pLvl,Dev):
        '''
        Evaluate the consumption function for this contract.
        '''
        cLvl, MedLvl, xLvl = self.__call__(mLvl,pLvl,Dev)
        return cLvl
    
    def MedFunc(self,mLvl,pLvl,Dev):
        '''
        Evaluate the medical care function for this contract.
        '''
        cLvl, MedLvl, xLvl = self.__call__(mLvl,pLvl,Dev)
        return MedLvl
    
    def xFunc(self,mLvl,pLvl,Dev):
        '''
        Evaluate the total expenditure function for this contract.
        '''
        cLvl, MedLvl, xLvl = self.__call__(mLvl,pLvl,Dev)
        return xLvl
        

    def evalvAndvPandvPP(self,mLvl,pLvl,Dev):
        '''
        Evaluate the value, marginal value, and marginal marginal value for this contract.
        
        Parameters
        ----------
        mLvl : np.array
             Market resource levels.
        pLvl : np.array
             Permanent income levels.
        Dev : np.array
             Medical need shocks.
             
        Returns
        -------
        v : np.array
            Value at each point in the input.
        vP : np.array
            Marginal value (with respect to market resources) at each point in the input.
        vPP : np.array
            Marginal marginal value (with respect to market resources) at each point in the input.
        '''
        if mLvl.shape is ():
            mLvl = np.array([mLvl])
            pLvl = np.array([pLvl])
            Dev = np.array([Dev])
            float_in = True
        else:
            float_in = False
            
        # Initialize output
        v = np.zeros_like(mLvl)
        vP = np.zeros_like(mLvl)
        vPP = np.zeros_like(mLvl)
            
        # Determine which inputs have zero medical shock and fill in output
        ZeroShk = np.logical_and(np.isinf(Dev), Dev < 0.)
        v[ZeroShk] = self.ValueFuncZeroShk(mLvl[ZeroShk],pLvl[ZeroShk])
        vP[ZeroShk] = self.cFuncZeroShk(mLvl[ZeroShk],pLvl[ZeroShk])**(-self.CRRA)
        
        # Get value of paying full price or paying "option cost" to pay coinsurance rate
        these = np.logical_not(ZeroShk)
        mTemp = mLvl[these]
        mTempAlt = mTemp-self.OptionCost
        pTemp = pLvl[these]
        DevTemp = Dev[these]
        v_Copay = self.ValueFuncCopay(mTempAlt,pTemp,DevTemp)
        v_Copay[mTempAlt < 0.] = -np.inf
        if self.OptionCost > 0.:
            v_FullPrice = self.ValueFuncFullPrice(mTemp,pTemp,DevTemp)
        
        # Decide which option is better and initialize output
        if self.OptionCost > 0.:
            Copay_better = v_Copay > v_FullPrice
        else:
            Copay_better = np.ones_like(v_Copay,dtype=bool)
        FullPrice_better = np.logical_not(Copay_better)
        
        # Fill in output using better of two choices
        cTemp = np.zeros_like(mTemp)
        vTemp = np.zeros_like(mTemp)
        vTemp[Copay_better] = self.ValueFuncCopay(mTempAlt[Copay_better],pTemp[Copay_better],DevTemp[Copay_better])
        vTemp[FullPrice_better] = self.ValueFuncFullPrice(mTemp[FullPrice_better],pTemp[FullPrice_better],DevTemp[FullPrice_better])
        cTemp[Copay_better], trash1, trash2 = self.PolicyFuncCopay(mTempAlt[Copay_better],pTemp[Copay_better],DevTemp[Copay_better])
        cTemp[FullPrice_better], trash1, trash2 = self.PolicyFuncFullPrice(mTemp[FullPrice_better],pTemp[FullPrice_better],DevTemp[FullPrice_better])
        v[these] = vTemp
        vP[these] = cTemp**(-self.CRRA)
        
        if float_in:
            return v[0], vP[0], vPP[0]
        else:
            return v, vP, vPP
        
        
class cAndMedFunc(HARKobject):
    '''
    A class representing the consumption and medical care function for one coinsurance
    rate in one discrete state. Its call function returns cLvl andMedLvl, iLvl.
    Also has functions for these controls individually.
    '''
    def __init__(self, xFunc, bFromxFunc, MedShkAvg, MedShkStd, EffPrice):
        '''
        Constructor method for a new instance of cAndMedFunc.
        
        Parameters
        ----------
        xFunc : function
            Expenditure function (cLvl & MedLvl), defined over (mLvl,pLvl,Dev).
        bFromxFunc : function
            Transformed consumption share function, defined over (xLvl,MedShkAdj). Badly named.
        MedShkMean : float
            Mean of log medical need shocks.
        MedShkStd : function
            Stdev of log medical need shocks.
        EffPrice : float
            Relative price of a unit of medical care: EffPrice = MedPrice*Copay.
        
        Returns
        -------
        None
        '''
        self.xFunc = xFunc
        self.bFromxFunc = bFromxFunc
        self.MedShkAvg = MedShkAvg
        self.MedShkStd = MedShkStd
        self.EffPrice = EffPrice
        
    def __call__(self,mLvl,pLvl,Dev):
        '''
        Evaluates the policy function and returns cLvl and MedLvl.
        
        Parameters
        ----------
        mLvl : np.array
            Array of market resources values.
        pLvl : np.array
            Array of permanent income levels.
        Dev : np.array
            Array of standard deviations from mean medical need shock.
         
        Returns
        -------
        cLvl : np.array
            Array of consumption levels.
        MedLvl : np.array
            Array of medical care levels.
        '''
        MedShk = np.exp(self.MedShkAvg + self.MedShkStd*Dev)
        xLvl = self.xFunc(mLvl,pLvl,Dev)
        
        cShareTrans = self.bFromxFunc(xLvl,MedShk*self.EffPrice)
        q = np.exp(-cShareTrans)
        cLvl = xLvl/(1.+q)
        MedLvl = xLvl/self.EffPrice*q/(1.+q)
        return cLvl, MedLvl, xLvl
        
    def cFunc(self,mLvl,pLvl,Dev):
        cLvl, MedLvl, xLvl = self(mLvl,pLvl,Dev)
        return cLvl
    
    def MedFunc(self,mLvl,pLvl,Dev):
        cLvl, MedLvl, xLvl = self(mLvl,pLvl,Dev)
        return MedLvl


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
        self.OptionCost = Deductible*(1.0-Copay)
        
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



def solveInsuranceSelection(solution_next,IncomeDstn,TaxFunc,SubsidyFunc,MedShkAvg,MedShkStd,ZeroMedShkPrb,MedShkCount,DevMin,DevMax,
                            LivPrb,DiscFac,CRRA,CRRAmed,BequestScale,BequestShift,Cfloor,Rfree,MedPrice,pLvlNextFunc,BoroCnstArt,aXtraGrid,
                            pLvlGrid,ContractList,HealthMrkvArray,ESImrkvFunc,ChoiceShkMag,EffPriceList,bFromxFunc,verbosity):
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
    TaxFunc : function
        Function that yields income taxes as a function of labor income.
    SubsidyFunc : [function]
       IMI subsidies as a function of mLvl, pLvl, and Premium (by health).
    MedShkAvg : [float]
        Mean value of log medical need shocks by health state.
    MedShkStd : [float]
        Standard deviation of log medical need shocks by health state.
    ZeroMedShkPrb : [float]
        Probability of having zero medical needs shock by health state.
    MedShkCount : int
        Number of (nonzero) medical need shocks in grid (grid is on Dev).
    DevMin : float
        Minimum number of standard deviations below MedShkAvg.
    DevMax : float
        Maximum number of standard deviations above MedShkAvg.
    LivPrb : np.array
        Survival probability; likelihood of being alive at the beginning of
        the succeeding period conditional on current health state.   
    DiscFac : float
        Intertemporal discount factor for future utility.        
    CRRA : float
        Coefficient of relative risk aversion for consumption.
    CRRAmed : float
        Coefficient of relative risk aversion for medical care.
    BequestShift : float
        Shifter in bequest motive function.
    BequestScale : float
        Scale of bequest motive function.
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
    HealthMrkvArray : numpy.array
        An NxN array representing a Markov transition matrix among discrete health
        states conditional on survival.  The i,j-th element of HealthMrkvArray
        is the probability of moving from state i in period t to state j in period t+1.
    ESImrkvFunc : numpy.array
        Function that returns a KxL array representing a Markov transition matrix
        among discrete ESI states, conditional on pLvl.  The i,j-th element of
        ESImrkvArray(pLvl) is the probability of moving from state i in period t
        to state j in period t+1.  Can take vector inputs, returns 3D output.
    ChoiceShkMag : float
        Magnitude of T1EV preference shocks for each insurance contract when making selection.
        Shocks are applied to pseudo-inverse value of contracts.
    EffPriceList : [float]
        Set of coinsurance rates across all contracts in ContractList, including 1.0 (full price).
    bFromxFunc : function
        Transformed consumption share as a function of total expenditure and effective
        medical need shock.
    verbosity : int
        How much output to print to screen during solution.  If solve is being called
        inside of a joblib spawned process, this must be 0.
                    
    Returns
    -------
    solution : InsuranceSelectionSolution
    '''
    t_start = clock()
    t0 = clock()
    
    # Get sizes of arrays
    mLvl_aug_factor = 3
    MedShk_aug_factor = 3
    pLvl_aug_factor = 4
    pLvlCount = pLvlGrid.size
    aLvlCount = aXtraGrid.size
    
    # Construct MrkvArray by pLvl by combining HealthMrkvArray and ESImrkvArray
    ESImrkvArray_all = ESImrkvFunc(pLvlGrid)
    MrkvArray_temp = combineIndepMrkvArrays(ESImrkvArray_all[0,:,:],HealthMrkvArray) # Example MrkvArray to get state counts
    StateCountNow  = MrkvArray_temp.shape[0] # number of discrete states this period
    StateCountNext = MrkvArray_temp.shape[1] # number of discrete states next period
    MrkvArray_all = np.zeros((pLvlCount,StateCountNow,StateCountNext)) + np.nan
    for j in range(pLvlCount):
        MrkvArray_all[j,:,:] = combineIndepMrkvArrays(ESImrkvArray_all[j,:,:],HealthMrkvArray)
    
    # Make regular and dense grids of Dev
    DevGrid = np.linspace(DevMin,DevMax,MedShkCount)
    DevGridDense = makeGridDenser(DevGrid,MedShk_aug_factor)
    
    # Make dense grid for pLvl (only used when MedShk=0)
    pGridDense = makeGridDenser(pLvlGrid,pLvl_aug_factor)
    
    # Make alternate versions of pLvlGrid and pGridDense
    aXtraMax = np.max(aXtraGrid)
    aLvlMax_min_acceptable = 10.
    pLvlAlt = aLvlMax_min_acceptable/aXtraMax
    aLvlMax_by_pLvl = aXtraMax*pLvlGrid
    pLvlGrid_alt = copy(pLvlGrid)
    pLvlGrid_alt[aLvlMax_by_pLvl < aLvlMax_min_acceptable] = pLvlAlt
    aLvlMax_by_pLvl = aXtraMax*pGridDense
    pGridDense_alt = copy(pGridDense)
    pGridDense_alt[aLvlMax_by_pLvl < aLvlMax_min_acceptable] = pLvlAlt
    
    # Make a JDfixer instance to use when necessary
    mGridDenseBase = makeGridDenser(aXtraGrid,mLvl_aug_factor)
    MyJDfixer = JDfixer(aLvlCount+1,MedShkCount,mGridDenseBase.size,DevGridDense.size)
    MyJDfixerZeroShk = JDfixerSimple(aLvlCount+1,pLvlGrid.size,mGridDenseBase.size,pGridDense.size,CRRA)
    
    # Define utility function derivatives and inverses
    u = lambda x : utility(x,CRRA)
    uMed = lambda x : utility(x,CRRAmed)
    uP = lambda x : utilityP(x,CRRA)
    uPinv = lambda x : utilityP_inv(x,CRRA)
    uinvP = lambda x : utility_invP(x,CRRA)
    uinv = lambda x : utility_inv(x,CRRA)
    BequestMotive = lambda x : BequestScale*utility(x+BequestShift,CRRA)
    BequestMotiveP = lambda x : BequestScale*utilityP(x+BequestShift,CRRA)
    
    # Get minimum value of next period's permanent income grid
    try:
        pLvlNext_min = solution_next.hLvl[0].x_list[1] # Element 0 is always 0.0, inserted point
    except:
        pLvlNext_min = 0.0 # Only in period T    
    
    # For each future health state, find the minimum allowable end of period assets by permanent income    
    aLvlMinCond = np.zeros((pLvlCount,StateCountNext))
    for h in range(StateCountNext):
        h_alt = np.mod(h,5)
        
        # Unpack the inputs conditional on future health state
        PermShkValsNext  = IncomeDstn[h_alt][1]
        TranShkValsNext  = IncomeDstn[h_alt][2]
        IncShkCount = PermShkValsNext.size
        PermShkVals_tiled = np.tile(np.reshape(PermShkValsNext,(1,IncShkCount)),(pLvlCount,1))
        TranShkVals_tiled = np.tile(np.reshape(TranShkValsNext,(1,IncShkCount)),(pLvlCount,1))
        pLvlGrid_tiled = np.tile(np.reshape(pLvlGrid,(pLvlCount,1)),(1,IncShkCount))
        pLvlNext = np.maximum(pLvlNextFunc(pLvlGrid_tiled)*PermShkVals_tiled,pLvlNext_min)
        yLvlNext = pLvlNext*TranShkVals_tiled
        aLvlMin_cand = (solution_next.mLvlMin(pLvlNext) - (yLvlNext - TaxFunc(yLvlNext)))/Rfree[h_alt]
        aLvlMinCond[:,h] = np.min(aLvlMin_cand,axis=1)
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
    EndOfPrdvCond = np.zeros((pLvlCount,aLvlCount+1,StateCountNext))
    EndOfPrdvPcond = np.zeros((pLvlCount,aLvlCount+1,StateCountNext))
    hLvlCond = np.zeros((pLvlCount,StateCountNext))
    for h in range(StateCountNext):
        h_alt = np.mod(h,5)
        
        # Unpack the inputs conditional on future health state
        ShkPrbsNext      = IncomeDstn[h_alt][0]
        PermShkValsNext  = IncomeDstn[h_alt][1]
        TranShkValsNext  = IncomeDstn[h_alt][2]
        ShkCount         = PermShkValsNext.size
        vPfuncNext       = solution_next.vPfunc[h]      
        vFuncNext        = solution_next.vFunc[h]
        
        # Calculate human wealth conditional on achieving this future health state
        PermIncNext   = np.tile(pLvlNextFunc(pLvlGrid),(ShkCount,1))*np.tile(PermShkValsNext,(pLvlCount,1)).transpose()
        PermIncNext   = np.maximum(PermIncNext,pLvlNext_min) 
        hLvlCond[:,h] = 1.0/Rfree[h_alt]*np.sum((np.tile(TranShkValsNext,(pLvlCount,1)).transpose()*PermIncNext + solution_next.hLvl[h](PermIncNext))*np.tile(ShkPrbsNext,(pLvlCount,1)).transpose(),axis=0)
        
        # Make arrays of current end of period states
        aNrmGrid    = np.insert(aXtraGrid,0,0.0)
        pLvlNow     = np.tile(pLvlGrid,(aLvlCount+1,1)).transpose()
        pLvlNow_alt = np.tile(pLvlGrid,(aLvlCount+1,1)).transpose()
        aLvlNow     = np.tile(aNrmGrid,(pLvlCount,1))*pLvlNow_alt
        pLvlNow_tiled = np.tile(pLvlNow,(ShkCount,1,1))
        aLvlNow_tiled = np.tile(aLvlNow,(ShkCount,1,1)) # shape = (ShkCount,pLvlCount,aLvlCount)
        if pLvlGrid[0] == 0.0:  # aLvl turns out badly if pLvl is 0 at bottom
            aLvlNow[0,:] = aNrmGrid*pLvlGrid[1]
            aLvlNow_tiled[:,0,:] = np.tile(aNrmGrid*pLvlGrid[1],(ShkCount,1))
            
        # Tile arrays of the income shocks and put them into useful shapes
        PermShkVals_tiled = np.transpose(np.tile(PermShkValsNext,(aLvlCount+1,pLvlCount,1)),(2,1,0))
        TranShkVals_tiled = np.transpose(np.tile(TranShkValsNext,(aLvlCount+1,pLvlCount,1)),(2,1,0))
        ShkPrbs_tiled     = np.transpose(np.tile(ShkPrbsNext,(aLvlCount+1,pLvlCount,1)),(2,1,0))
        
        # Make grids of future states conditional on achieving this future health state
        pLvlNext = pLvlNextFunc(pLvlNow_tiled)*PermShkVals_tiled
        pLvlNext = np.maximum(pLvlNext, pLvlNext_min)
        yLvlNext = pLvlNext*TranShkVals_tiled
        mLvlNext = Rfree[h_alt]*aLvlNow_tiled + yLvlNext - TaxFunc(yLvlNext)
        
        # Calculate future-health-conditional end of period value and marginal value
        tempv = vFuncNext(mLvlNext,pLvlNext)
        tempvP = vPfuncNext(mLvlNext,pLvlNext)
        EndOfPrdvPcond[:,:,h]  = Rfree[h_alt]*np.sum(tempvP*ShkPrbs_tiled,axis=0)
        EndOfPrdvCond[:,:,h]   = np.sum(tempv*ShkPrbs_tiled,axis=0)
        
#        temp = np.isnan(tempv)
#        if np.any(temp):
#            print('vNext',h,np.sum(temp))
        
    # Calculate end of period value and marginal value conditional on each current health state
    EndOfPrdv = np.zeros((pLvlCount,aLvlCount+1,StateCountNow))
    EndOfPrdvP = np.zeros((pLvlCount,aLvlCount+1,StateCountNow))
    for h in range(StateCountNow):
        h_alt = np.mod(h,5) # Alternate state index, very hacky
        
        # Set up a temporary health transition array
        MrkvArray_temp = np.tile(np.reshape(MrkvArray_all[:,h,:],(pLvlCount,1,StateCountNext)),(1,aLvlCount+1,1))
        DiscFacEff = DiscFac*LivPrb[h_alt] # "effective" discount factor
        DiePrb = (1.-LivPrb[h_alt])
        
        # Weight future health states according to the transition probabilities
        EndOfPrdv[:,:,h]  = DiscFacEff*np.sum(EndOfPrdvCond*MrkvArray_temp,axis=2) + DiePrb*BequestMotive(aLvlNow)
        EndOfPrdvP[:,:,h] = DiscFacEff*np.sum(EndOfPrdvPcond*MrkvArray_temp,axis=2) + DiePrb*BequestMotiveP(aLvlNow)
        
        temp = np.isnan(EndOfPrdv[:,:,h])
        if np.any(temp):
            print('EndOfPrdv',h,np.sum(temp))
            
    # Calculate human wealth conditional on each current health state 
    hLvlGrid = np.sum(MrkvArray_all*np.tile(np.reshape(hLvlCond,(pLvlCount,1,StateCountNext)),(1,StateCountNow,1)),axis=2)
    for h in range(StateCountNow):
        h_alt = np.mod(h,5)
        hLvlGrid[:,h_alt] *= LivPrb[h_alt]
    if np.all(LivPrb == 0.):
        hLvlGrid[:,:] = 0.
    
    t1 = clock()
    FutureExpectations_time = t1 - t0
    
    # Loop through current health states to solve the period at each one
    solution_now = InsuranceSelectionSolution(mLvlMin=mLvlMinNow)
    JDfixCount = 0
    SolutionConstruction_time = 0.
    MedShkIntegration_time = 0.
    UpperEnvelope_time = 0.
    for h in range(StateCountNow):
        t0 = clock() # Beginning of solution construction step
        h_alt = np.mod(h,5)
        
        mCount           = EndOfPrdvP.shape[1]
        pCount           = EndOfPrdvP.shape[0]
        MedShkVals       = np.exp(MedShkAvg[h_alt] + MedShkStd[h_alt]*DevGrid)
        PolicyFuncsThisHealthCopay = []
        vFuncsThisHealthCopay = []
         
        # Make the end of period value function for this health state
        EndOfPrdvNvrsFunc_by_pLvl = []
        EndOfPrdvNvrs = uinv(EndOfPrdv[:,:,h])
        for j in range(pLvlCount):
            a_temp = aLvlNow[j,:]
            EndOfPrdvNvrs_temp = EndOfPrdvNvrs[j,:]
            EndOfPrdvNvrsFunc_by_pLvl.append(LinearInterp(a_temp,EndOfPrdvNvrs_temp))
        EndOfPrdvNvrsFuncBase = LinearInterpOnInterp1D(EndOfPrdvNvrsFunc_by_pLvl,pLvlGrid)
        EndOfPrdvNvrsFunc = EndOfPrdvNvrsFuncBase
        EndOfPrdvFunc = ValueFunc2D(EndOfPrdvNvrsFunc,CRRA)
        EndOfPrdvNvrsFunc_Cnst = LinearInterp(pLvlGrid,uinv(EndOfPrdv[:,0,h]))
        EndOfPrdvFunc_Cnst = ValueFunc(EndOfPrdvNvrsFunc_Cnst,CRRA)
        
        # Make tiled versions of end-of-period marginal value and medical needs shocks
        EndOfPrdvPnvrs = uPinv(EndOfPrdvP[:,:,h])
        EndOfPrdvPnvrs_tiled = np.tile(np.reshape(EndOfPrdvPnvrs,(1,pCount,mCount)),(MedShkCount,1,1))
        MedShkVals_tiled  = np.tile(np.reshape(MedShkVals,(MedShkCount,1,1)),(1,pCount,mCount))
        DevGrid_tiled  = np.tile(np.reshape(DevGrid,(MedShkCount,1,1)),(1,pCount,mCount))
        
        # Make consumption, value, and marginal value functions for when there is zero medical need shock
        cLvl_data = EndOfPrdvPnvrs.transpose()
        mLvl_data = aLvlNow.transpose() + cLvl_data
        pLvl_data = pLvlNow.transpose()
        v_data = u(cLvl_data) + EndOfPrdv[:,:,h].transpose()
        vNvrs_data = uinv(v_data)
        cFuncZeroShk, vFuncZeroShk = MyJDfixerZeroShk(mLvl_data,pLvl_data,vNvrs_data,cLvl_data,mGridDenseBase,pGridDense,pGridDense_alt,EndOfPrdvFunc_Cnst)
        
#        if h == 7:
#            print(mLvl_data[-1,0],mLvl_data[-1,1])
            
        # For each coinsurance rate, make policy and value functions (for this health state)
        for k in range(len(EffPriceList)):
            MedPriceEff = EffPriceList[k]
            
            # Calculate endogenous gridpoints and controls
            cLvlNow = EndOfPrdvPnvrs_tiled
            MedLvlNow = MedPriceEff**(-1./CRRAmed)*MedShkVals_tiled**(1.-1./CRRAmed)*cLvlNow**(CRRA/CRRAmed)
            xLvlNow = cLvlNow + MedLvlNow*MedPriceEff
            aLvlNow_tiled = np.tile(np.reshape(aLvlNow,(1,pCount,mCount)),(MedShkCount,1,1))
            mLvlNow = xLvlNow + aLvlNow_tiled
                
            # Determine which pLvls will need the G2EGM convexity fix
            NonMonotonic = np.any((mLvlNow[:,:,1:]-mLvlNow[:,:,:-1]) < 0.,axis=(0,2)) # whether each pLvl has non-monotonic pattern in mLvl gridpoints
            HasNaNs = np.any(np.isnan(xLvlNow),axis=(0,2)) # whether each pLvl contains any NaNs due to future vP=0.0
            NeedsJDfix = np.logical_or(NonMonotonic,HasNaNs)
            JDfixCount += np.sum(NeedsJDfix)
            
            # Loop over each permanent income level and medical shock and make an xFunc
            xFunc_by_pLvl = [] # Initialize the empty list of 2D xFuncs
            mNrmGrid = mGridDenseBase
            mCountAlt = mNrmGrid.size
            xLvlArray = np.zeros(((MedShkCount,pCount,mCountAlt)))
            for i in range(pCount):
                pLvl_i = pLvlGrid_alt[i]
                if pLvl_i == 0.0:
                    pLvl_i = pLvlGrid_alt[i+1]
                mGridDense = mGridDenseBase*pLvl_i
                if NeedsJDfix[i]:
                    aLvl_temp = aLvlNow_tiled[:,i,:]
                    pLvl_temp = pLvlGrid[i]*np.ones_like(aLvl_temp)
                    vNvrs_data = (uinv(u(cLvlNow[:,i,:]) + uMed(MedLvlNow[:,i,:]/MedShkVals_tiled[:,i,:]) + EndOfPrdvFunc(aLvl_temp,pLvl_temp))).transpose()
                    mLvl_data = mLvlNow[:,i,:].transpose()
                    xLvl_data = xLvlNow[:,i,:].transpose()
                    Dev_data = DevGrid_tiled[:,i,:].transpose()
                    xFunc_this_pLvl, xLvlArray[:,i,:] = MyJDfixer(mLvl_data,Dev_data,vNvrs_data,xLvl_data,mGridDense,DevGridDense)
                    xFunc_by_pLvl.append(xFunc_this_pLvl)
                    mLvlNow[:,i,0] = 0.0 # This fixes the "seam" problem so there are no NaNs
                else:
                    tempArray = np.zeros((mCountAlt,MedShkCount))
                    for j in range(MedShkCount):
                        m_temp = mLvlNow[j,i,:] - mLvlNow[j,i,0]
                        x_temp = xLvlNow[j,i,:]
                        xFunc_temp = LinearInterp(np.insert(m_temp,0,0.0),np.insert(x_temp,0,0.0))
                        tempArray[:,j] = xFunc_temp(mGridDense)
                        idx = np.searchsorted(mGridDense,mLvlNow[j,i,0])
                        xLvlArray[j,i,idx:] = xFunc_temp(mGridDense[idx:] - mLvlNow[j,i,0])
                        xLvlArray[j,i,:idx] = mGridDense[:idx]
                    xFunc_by_pLvl.append(BilinearInterp(tempArray,mGridDense,DevGrid))
                
            # Combine the many expenditure functions into a single one and adjust for the natural borrowing constraint
            xFuncNowUncBase = TwistFuncB(LinearInterpOnInterp2D(xFunc_by_pLvl,pLvlGrid))
            ConstraintSeam = BilinearInterp((mLvlNow[:,:,0]).transpose(),pLvlGrid,DevGrid)
            xFuncNowUnc = VariableLowerBoundFunc3Dalt(xFuncNowUncBase,ConstraintSeam)
            xFuncNow = CompositeFunc3D(xFuncNowCnst,xFuncNowUnc,ConstraintSeam)
            
            # Make a policy function for this coinsurance rate and health state
            PolicyFuncsThisHealthCopay.append(cAndMedFunc(xFuncNow, bFromxFunc, MedShkAvg[h_alt], MedShkStd[h_alt], MedPriceEff))
            
            # Calculate pseudo inverse value on a grid of states for this coinsurance rate
            pLvlArray = np.tile(np.reshape(pLvlGrid,(1,pCount,1)),(MedShkCount,1,mCountAlt))
            pLvlArray_alt = np.tile(np.reshape(pLvlGrid_alt,(1,pCount,1)),(MedShkCount,1,mCountAlt))
            mMinArray = np.tile(np.reshape(mLvlMinNow(pLvlGrid),(1,pCount,1)),(MedShkCount,1,mCountAlt))
            mLvlArray = mMinArray + np.tile(np.reshape(mNrmGrid,(1,1,mCountAlt)),(MedShkCount,pCount,1))*pLvlArray_alt
            if pLvlGrid[0] == 0.0:  # mLvl turns out badly if pLvl is 0 at bottom
                mLvlArray[:,0,:] = mLvlArray[:,1,:]
            MedShkArray = np.tile(np.reshape(MedShkVals,(MedShkCount,1,1)),(1,pCount,mCountAlt))
            DevArray = np.tile(np.reshape(DevGrid,(MedShkCount,1,1)),(1,pCount,mCountAlt))            
            cShareTrans = bFromxFunc(xLvlArray,MedShkArray*MedPriceEff)
            q = np.exp(-cShareTrans)
            cLvlArray = xLvlArray/(1.+q)
            MedLvlArray = xLvlArray/MedPriceEff*q/(1.+q)            
            aLvlArray = np.maximum(mLvlArray - xLvlArray,0.0) # OCCASIONAL VIOLATIONS BY 1E-18 !!!
            EndOfPrdvArray = EndOfPrdvFunc(aLvlArray,pLvlArray)
            vNow = u(cLvlArray) + uMed(MedLvlArray/MedShkArray) + EndOfPrdvArray
            vNvrsNow  = uinv(vNow)
            
#            temp = xLvlArray < 0.
#            if h==0 and np.any(temp):
#                print('vNvrs_contruction',h,k,np.sum(temp))
            
            # Calculate "candidate" v_ZeroShk values from consuming all expenditure with zero MedShk
            v_ZeroShk_cand = np.max(u(xLvlArray) + EndOfPrdvArray,axis=0)
            v_ZeroShk_orig = vFuncZeroShk(mLvlArray[0,:,:], pLvlArray[0,:,:])
            
            # Modify v_ZeroShk so that it is greater than all non-zero MedShk values (fixes small numeric problems)
            v_ZeroShk = np.maximum(v_ZeroShk_orig,v_ZeroShk_cand)
            vNvrs_ZeroShk = uinv(v_ZeroShk) # "Modified" vNvrs with zero medical need shock
            vNvrs_ZeroShk_tiled = np.tile(np.reshape(vNvrs_ZeroShk,(1,pCount,mCountAlt)),(MedShkCount,1,1))
            vNvrsScaled = np.log(vNvrs_ZeroShk_tiled/vNvrsNow - 1.) # "Rescaled" vNvrs relative to ZeroShk
            vNvrs_ZeroShk_plus = np.concatenate((np.zeros((pCount,1)),vNvrs_ZeroShk),axis=1)
            
            # Add the value function to the list for this health
            vFunc_this_health_copay = ValueFuncCL(mNrmGrid,pLvlGrid,pLvlGrid_alt,vNvrs_ZeroShk_plus,vNvrsScaled,CRRA,DevMin,DevMax,MedShkCount)
            vFuncsThisHealthCopay.append(vFunc_this_health_copay)
            
        t1 = clock() # End of solution construction step, beginning of MedShk integration step
            
        # Set up state grids to prepare for the medical shock integration step
        tempArray    = np.tile(np.reshape(aXtraGrid,(aLvlCount,1,1)),(1,pLvlCount,MedShkCount))
        mMinArray    = np.tile(np.reshape(mLvlMinNow(pLvlGrid),(1,pLvlCount,1)),(aLvlCount,1,MedShkCount))
        pLvlArray    = np.tile(np.reshape(pLvlGrid,(1,pLvlCount,1)),(aLvlCount,1,MedShkCount))
        pLvlArray_alt= np.tile(np.reshape(pLvlGrid_alt,(1,pLvlCount,1)),(aLvlCount,1,MedShkCount))
        mLvlArray    = mMinArray + tempArray*pLvlArray_alt + Cfloor
        if pLvlGrid[0] == 0.0: # Fix the problem of all mLvls = 0 when pLvl = 0
            mLvlArray[:,0,:] = mLvlArray[:,1,:]
            
        # Calculate value and marginal value when medical needs shock is zero
        vArrayZeroShk = vFuncZeroShk(mLvlArray[:,:,0],pLvlArray[:,:,0])
        vParrayZeroShk = cFuncZeroShk(mLvlArray[:,:,0],pLvlArray[:,:,0])**(-CRRA)
                
        # For each insurance contract available in this state, "integrate" across medical need
        # shocks to get policy and (marginal) value functions for each contract
        PolicyFuncsThisHealth = []
        vFuncsThisHealth = []
        vPfuncsThisHealth = []
        AVfuncsThisHealth = []
        CritDevFuncsThisHealth = []
        for z in range(len(ContractList[h])):
            # Set and unpack the contract of interest
            Contract = ContractList[h][z]
            Copay = Contract.Copay
            OptionCost = Contract.OptionCost
            FullPrice_idx = np.argwhere(np.array(EffPriceList)==MedPrice)[0][0]
            Copay_idx = np.argwhere(np.array(EffPriceList)==Copay*MedPrice)[0][0]
            
            # Get the value and policy functions for this contract
            vFuncFullPrice = vFuncsThisHealthCopay[FullPrice_idx]
            vFuncCopay = vFuncsThisHealthCopay[Copay_idx]
            PolicyFuncFullPrice = PolicyFuncsThisHealthCopay[FullPrice_idx]
            PolicyFuncCopay = PolicyFuncsThisHealthCopay[Copay_idx]
            
            # Find the critical med shock where Cfloor binds for each (mLvl,pLvl) value
            # Find CritDev using a clever closed formula
            m_temp = mLvlArray[:,:,0]
            p_temp = pLvlArray[:,:,0]
            m_temp_alt = m_temp - Contract.OptionCost
            LogCritShk = (CRRAmed - CRRA)/(CRRAmed - 1.)*np.log(Cfloor) + CRRAmed/(CRRAmed - 1.)*np.log(m_temp_alt/Cfloor - 1.) - np.log(MedPrice*Copay)
            LogCritShk[np.isnan(LogCritShk)] = -np.inf
            CritDevArray = (LogCritShk - MedShkAvg[h_alt])/MedShkStd[h_alt]
            CritDevArray = np.maximum(CritDevArray,DevMin) # Apply lower bound for grid
            CritDevFunc_by_pLvl = []
            for j in range(pCount):
                CritDevFunc_by_pLvl.append(LinearInterp(np.insert(m_temp[:,j],0,0.0),np.insert(CritDevArray[:,j],0,DevMin)))
            CritDevFunc = LinearInterpOnInterp1D(CritDevFunc_by_pLvl,pLvlGrid)            
            Always_Cfloor = np.logical_or(CritDevArray == DevMin, m_temp <= Cfloor) # Critical shock happens lower than lowest shock in grid (-3std)
            CritShkPrbArray = norm.sf(CritDevArray)
            CritShkPrbArray[Always_Cfloor] = 1.
            
            # Make the policy function for this contract
            PolicyFuncsThisHealth.append(InsSelPolicyFunc(vFuncZeroShk,vFuncFullPrice,vFuncCopay,cFuncZeroShk,PolicyFuncFullPrice,PolicyFuncCopay,Contract,CRRA))
            CritDevFuncsThisHealth.append(CritDevFunc)
            
            # Choose medical need shock grids for integration
            FracArray = np.tile(np.reshape(np.linspace(0.0,1.0,MedShkCount),(1,1,MedShkCount)),(aLvlCount,pLvlCount,1))
            DevArray = DevMin + FracArray*(np.tile(np.reshape(np.minimum(CritDevArray,DevMax)-DevMin,(aLvlCount,pLvlCount,1)),(1,1,MedShkCount)))
    
            # Calculate probabilities of all of the medical shocks
            BasePrbArray = norm.pdf(DevArray)
            SumPrbArray = np.sum(BasePrbArray,axis=2)
            AdjArray = np.tile(np.reshape((1.0-ZeroMedShkPrb[h_alt])*(1.0-CritShkPrbArray)/SumPrbArray,(aLvlCount,pLvlCount,1)),(1,1,MedShkCount))
            MedShkPrbArray = BasePrbArray*AdjArray
                           
            # Get value and marginal value at an array of states
            vArrayBig, vParrayBig, vPParrayBig = PolicyFuncsThisHealth[-1].evalvAndvPandvPP(mLvlArray,pLvlArray,DevArray)
            ConArrayBig, MedArrayBig, xLvlArrayBig = PolicyFuncsThisHealth[-1](mLvlArray,pLvlArray,DevArray)
            
            #temp = np.isnan(vArrayBig)
            #if np.any(temp) and h==0:
            #    print('vBig',h,z,np.argwhere(temp))
            
            # Calculate expected value when hitting Cfloor (use truncated lognormal formula)
            mu = (1.-1./CRRAmed)*MedShkAvg[h_alt] # underlying mean of lognormal shocks
            sigma = (1.-1./CRRAmed)*MedShkStd[h_alt] # underlying std of lognormal shocks
            C1 = (MedPrice*Copay)**(1.-1./CRRAmed)*Cfloor**(CRRA/CRRAmed - CRRA)/(1.-CRRAmed) # constant factor
            ExpectedAdjShkAtCfloor = -0.5*np.exp(mu+(sigma**2)*0.5)*(erfc(np.sqrt(0.5)*(sigma-CritDevArray))-2.0)/CritShkPrbArray
            ExpectedUMedatCfloor = ExpectedAdjShkAtCfloor*C1
            vFloor_expected = u(Cfloor) + EndOfPrdvFunc(np.zeros_like(m_temp),p_temp) + ExpectedUMedatCfloor
                
            # Calculate value for portions of the state space where mLvl <= Cfloor, so Cfloor always binds
            vFloorBypLvl = u(Cfloor) + C1*np.exp(mu+(sigma**2)*0.5) + EndOfPrdvFunc(np.zeros_like(pLvlGrid),pLvlGrid)
            vNvrsFloorBypLvl = uinv(vFloorBypLvl)
            vFloor_tiled = np.tile(np.reshape(vFloorBypLvl,(1,pLvlCount,1)),(aLvlCount,1,MedShkCount))
            vArrayBig = np.maximum(vArrayBig,vFloor_tiled) # This prevents tiny little non-monotonicities in vFunc
            if z == 0:
                vNvrsFloorBypLvl_default = vNvrsFloorBypLvl # This is for the "default" or "null" or "free" contract
            
            # Integrate (marginal) value across medical shocks
            vArrayMain = np.sum(vArrayBig*MedShkPrbArray,axis=2)
            vArray   = vArrayMain + ZeroMedShkPrb[h_alt]*vArrayZeroShk + (1.0-ZeroMedShkPrb[h_alt])*CritShkPrbArray*vFloor_expected
            vParray  = np.sum(vParrayBig*MedShkPrbArray,axis=2) + ZeroMedShkPrb[h_alt]*vParrayZeroShk + (1.0-ZeroMedShkPrb[h_alt])*CritShkPrbArray*0.0
            
#            temp = np.isnan(vArrayZeroShk)
#            if np.any(temp):
#                print('vZeroShk',h,z,np.sum(temp))
            
            # Calculate actuarial value at each (mLvl,pLvl), combining shocks above and below the critical value
            AVarrayBig = MedArrayBig*MedPrice - Contract.OOPfunc(MedArrayBig) # realized "actuarial value" below critical shock
            ExpectedMedatCfloor = (Copay*MedPrice)**(-1./CRRAmed)*ExpectedAdjShkAtCfloor*Cfloor**(CRRA/CRRAmed) # Use truncated lognormal formula
            AVarray  = np.sum(AVarrayBig*MedShkPrbArray,axis=2) + (1.0-ZeroMedShkPrb[h_alt])*CritShkPrbArray*(ExpectedMedatCfloor*(1.-Copay) - OptionCost)
            
            # Construct pseudo-inverse arrays of vNvrs and vPnvrs, adding some data at the bottom
            mLvlArray_temp = mLvlArray[:,:,0]
            mLvlBound = mLvlArray_temp[0,:] # Lowest mLvl for each pLvl; will be used to make "seam"
            mGrid_small_A = np.concatenate((np.zeros((1,pLvlCount)),Cfloor*np.ones((1,pLvlCount)),mLvlArray_temp),axis=0) # for value
            mGrid_small_B = mLvlArray_temp - np.tile(np.reshape(mLvlBound,(1,pLvlCount)),(aLvlCount,1)) # for marginal value
            mGrid_small_C = np.concatenate((np.zeros((1,pLvlCount)),mLvlArray_temp),axis=0) # for AV
            vPnvrsArray   = uPinv(vParray)
            vNvrsArray    = np.concatenate((np.tile(np.reshape(vNvrsFloorBypLvl,(1,pLvlCount)),(2,1)),uinv(vArray)),axis=0)
            
#            temp = np.isnan(vNvrsArray)
#            if np.any(temp):
#                print('vNvrs',h,z,np.sum(temp))
            
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
            
        t2 = clock() # End of MedShk integration step, beginning of upper envelope step
        
        # If there is only one contract, then value and marginal value functions are trivial.
        if len(ContractList[h]) == 1:
            vFunc = vFuncsThisHealth[0]   # only element
            vPfunc = vPfuncsThisHealth[0] # only element
        
        else: # If there is more than one contract available, take expectation over choice shock.
            mLvlGrid = mGridDenseBase
            mLvlCount = mLvlGrid.size
        
            # Make grids to prepare for the choice shock step
            tempArray    = np.tile(np.reshape(mLvlGrid,(mLvlCount,1)),(1,pLvlCount))
            mMinArray    = np.tile(np.reshape(mLvlMinNow(pLvlGrid),(1,pLvlCount)),(mLvlCount,1))
            pLvlArray    = np.tile(np.reshape(pLvlGrid,(1,pLvlCount)),(mLvlCount,1))
            pLvlArray_alt= np.tile(np.reshape(pLvlGrid_alt,(1,pLvlCount)),(mLvlCount,1))
            mLvlArray    = mMinArray + tempArray*pLvlArray_alt + Cfloor
            if pLvlGrid[0] == 0.0: # Fix the problem of all mLvls = 0 when pLvl = 0
                mLvlArray[:,0] = mLvlArray[:,1]
            
            # Get value and marginal value at each point in mLvl x pLvl at each contract, taking account of premiums paid
            vArrayBig   = np.zeros((mLvlCount,pLvlCount,len(ContractList[h]))) # value at each gridpoint for each contract
            vParrayBig  = np.zeros((mLvlCount,pLvlCount,len(ContractList[h]))) # marg value at each gridpoint for each contract
            AdjusterArray = np.zeros((mLvlCount,pLvlCount,len(ContractList[h]))) # (1 - dPremium/dmLvl), usually equals 1
            UnaffordableArray = np.zeros((mLvlCount,pLvlCount,len(ContractList[h])),dtype=bool)
            for z in range(len(ContractList[h])):
                Contract = ContractList[h][z]
                PremiumArray = Contract.Premium(mLvlArray,pLvlArray)
                if (z > 0) and (h < 5): # Subsidy and adjuster are complicated
                    SubsidyArray = SubsidyFunc[h](mLvlArray,pLvlArray,PremiumArray)
                    excess_subsidy = SubsidyArray > PremiumArray
                    SubsidyArray[excess_subsidy] = PremiumArray[excess_subsidy]
                    dPremdm = Contract.Premium.derivativeX(mLvlArray,pLvlArray)
                    Adjuster = 1.0 - dPremdm + SubsidyFunc[h].derivativeX(mLvlArray,pLvlArray,PremiumArray) + dPremdm*SubsidyFunc[h].derivativeZ(mLvlArray,pLvlArray,PremiumArray)
                    Adjuster[excess_subsidy] = 1.0
                    AdjusterArray[:,:,z] = Adjuster
                else: # If not IMI, then subsidy and adjuster are easy
                    SubsidyArray = np.zeros_like(PremiumArray)
                    dPremdm = Contract.Premium.derivativeX(mLvlArray,pLvlArray)
                    AdjusterArray[:,:,z] = 1.0 - dPremdm
                mLvlArray_temp = mLvlArray - (PremiumArray - SubsidyArray)
                UnaffordableArray[:,:,z] = mLvlArray_temp <= 0
                vArrayBig[:,:,z]   = vFuncsThisHealth[z](mLvlArray_temp,pLvlArray)
                vParrayBig[:,:,z]  = vPfuncsThisHealth[z](mLvlArray_temp,pLvlArray)
                     
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
                z_idx = np.argmax(vNvrsArrayBig,axis=2)
                m_idx = np.tile(np.reshape(np.arange(mLvlCount),(mLvlCount,1)),(1,pLvlCount))
                p_idx = np.tile(np.reshape(np.arange(pLvlCount),(1,pLvlCount)),(mLvlCount,1))
                vNvrsArray  = vNvrsArrayBig[m_idx,p_idx,z_idx]
                vPnvrsArray = vPnvrsArrayBig[m_idx,p_idx,z_idx]
                                        
            # Make value and marginal value functions for the very beginning of the period, before choice shocks are drawn
            vNvrsArray_plus = np.concatenate((np.tile(np.reshape(vNvrsFloorBypLvl_default,(1,pLvlCount)),(2,1)),vNvrsArray),axis=0)
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
                                        policyFunc=PolicyFuncsThisHealth,vFuncByContract=vFuncsThisHealth,
                                        CritDevFunc=CritDevFuncsThisHealth)
        t3 = clock() # End of upper envelope step
        
        SolutionConstruction_time += (t1-t0)
        MedShkIntegration_time += (t2-t1)
        UpperEnvelope_time += (t3-t2)
    
    # Return the solution for this period
    t_end = clock()
    if verbosity > 0:
        print('Solving a period of the problem took ' + str(t_end-t_start) + ' seconds, fix count = ' + str(JDfixCount))
    if verbosity > 5:
        print('Computing expectations over income shocks took ' + str(FutureExpectations_time) + ' seconds.')
        print('Constructing policy functions for each health-copay took ' + str(SolutionConstruction_time) + ' seconds.')
        print('Integrating over MedShk for each health-contract took ' + str(MedShkIntegration_time) + ' seconds.')
        print('Finding the upper envelope among contracts took ' + str(FutureExpectations_time) + ' seconds.\n')
        
    return solution_now
    
####################################################################################################
    
class InsSelConsumerType(MedShockConsumerType,MarkovConsumerType):
    '''
    Class for representing consumers in the insurance selection model.  Each period, they receive
    shocks to their discrete health state and permanent and transitory income; after choosing an
    insurance contract, they learn their medical need shock and choose levels of consumption and
    medical care.
    '''
    _time_vary = ['DiscFac','LivPrb','MedPrice','ContractList','HealthMrkvArray','ESImrkvFunc','ChoiceShkMag','MedShkAvg','MedShkStd','ZeroMedShkPrb','SubsidyFunc']
    _time_inv = ['CRRA','CRRAmed','BequestScale','BequestShift','Cfloor','Rfree','BoroCnstArt','MedShkCount','DevMin','DevMax','TaxFunc','verbosity']
    
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
                           cycles=cycles,time_flow=time_flow,pseudo_terminal=True,**kwds)
        self.solveOnePeriod = solveInsuranceSelection # Choose correct solver
        self.poststate_vars = ['aLvlNow']
        self.time_vary = copy(InsSelConsumerType._time_vary)
        self.time_inv = copy(InsSelConsumerType._time_inv)

 
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
        for h in range(self.HealthMrkvArray[0].shape[0]):
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
        nu  = self.MedCurve*rho
        self.bFromxFunc = TransConShareFunc(rho,nu)
        self.CRRAmed = nu
        self.addToTimeInv('CRRAmed','bFromxFunc')
        
        
    def updateEffPriceList(self):
        '''
        Constructs the time-varying attribute EffPriceList.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        None
        '''
        orig_time = self.time_flow
        self.timeFwd()
        
        EffPriceList = [] # Effective medical care price for each period in the cycle
        for t in range(self.T_cycle):
            MedPrice = self.MedPrice[t]
            EffPriceList_temp = [MedPrice]
            for h in range(len(self.ContractList[t])):
                for Contract in self.ContractList[t][h]:
                    EffPriceList_temp.append(Contract.Copay*MedPrice)
            EffPriceList.append(np.unique(np.array(EffPriceList_temp)))
            
        self.EffPriceList = EffPriceList
        self.addToTimeVary('EffPriceList')
        
        if not orig_time:
            self.timeRev()
            

    def installPremiumFuncs(self):
        '''
        Applies the premium data in the attribute PremiumFuncs to the contracts
        in the attribute ContractList, accounting for the employer contribution
        in EmpContr.  The premiums in PremiumFuncs are used in the static
        solver to find quasi-equilibrium premiums; this method feeds these back
        into ContractList to use on the next dynamic solution pass.  Also updates
        the attribute TaxFunc to have the newly updated tax rate.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        None
        '''
        # First, update the health tax rate
        self.TaxFunc.rate = self.HealthTaxRate
        
        if not hasattr(self,'PremiumFuncs'):
            return # Don't do anything if PremiumFuncs hasn't been defined
        
        time_orig = self.time_flow
        self.timeFwd()
        T_retire = 40 # For determining whether to apply subsidy
        T = len(self.PremiumFuncs)
        
        # Loop through each age-state-contract, filling in the updated premium in ContractList
        for t in range(T):
            #StateCount = len(self.ContractList[t])
            for h in range(5): # Do individual market or retired premiums
                ContractCount = len(self.ContractList[t][h])
                for z in range(ContractCount):
                    if z == 0:
                        if t < T_retire:
                            PremiumFunc = self.UninsuredPremiumFunc
                        else:
                            PremiumFunc = ConstantFunction(0.0)
                    else:
                        PremiumFunc = self.PremiumFuncs[t][h][z]
                    self.ContractList[t][h][z].Premium = PremiumFunc    
            
            if t < T_retire:
                for h in range(5,10): # Do ESI market with no employer contribution
                    ContractCount = len(self.ContractList[t][h])
                    for z in range(ContractCount):
                        if z == 0:
                            if self.MandateForESI:
                                PremiumFunc = self.UninsuredPremiumFunc
                            else: # MandateForESI indicates whether IM applies to ESI
                                PremiumFunc = ConstantFunction(0.0)
                        else:
                            PremiumFunc = self.PremiumFuncs[t][h][z]
                        self.ContractList[t][h][z].Premium = PremiumFunc
                            
                for h in range(10,15): # Do ESI market *with* employer contribution
                    ContractCount = len(self.ContractList[t][h])
                    for z in range(ContractCount):
                        if z == 0:
                            if self.MandateForESI:
                                PremiumFunc = self.UninsuredPremiumFunc
                            else: # MandateForESI indicates whether IM applies to ESI
                                PremiumFunc = ConstantFunction(0.0)
                        else:
                            PremiumFunc = self.PremiumFuncs[t][h][z]
                        if PremiumFunc.__class__.__name__ == 'ConstantFunction':
                            NetPremium = np.maximum(PremiumFunc.value - (t < T_retire)*self.EmpContr, 0.0)
                            self.ContractList[t][h][z].Premium = ConstantFunction(NetPremium)
                        else:
                            self.ContractList[t][h][z].Premium = PremiumFunc
                        
        # Restore the original flow of time
        if not time_orig:
            self.timeRev()
        
        
    def updateUninsuredPremium(self,MandateTaxRate=0.,MandateFloor=0.,MandateForESI=False):
        '''
        Create the attribute UninsuredPremiumFunc, a function that will be used
        as the "premium" for being uninsured.  It is installed automatically by
        installPremiumFuncs, which is called by preSolve.  Will make (effectively)
        a free uninsured plan by default, but can be used for individual mandate.
        
        Parameters
        ----------
        MandateTaxRate : float
            Percentage of permanent income that must be paid in order to be uninsured.
            Defaults to zero.  Actual "premium" never exceeds 50% of market resources.
        MandateFloor : float
            Minimum value of the individual mandate penalty.
        MandateForESI : bool
            Indicator for whether the individual mandate applies to those who are offered ESI.
            
        Returns
        -------
        None
        '''
        self.MandateForESI = MandateForESI
        if np.logical_and(MandateTaxRate == 0., MandateFloor == 0.):
            self.UninsuredPremiumFunc = ConstantFunction(0.)
            return # Handle null IM policy
        
        # Handle non-null IM policy
        X = MandateTaxRate # For easier typing
        mLvlGrid = np.array([0.,100.])
        pLvlGrid = np.array([0.,100.])
        # First make the penalty based on (permanent) income
        pLvlBasedPenalty = BilinearInterp(np.array([[0.,X*100.],[0.,X*100.]]),mLvlGrid,pLvlGrid)
        if MandateFloor > 0.:
            ConstantPenalty  = ConstantFunction(MandateFloor)
            MainPenalty = UpperEnvelope2D(pLvlBasedPenalty,ConstantPenalty)
        else:
            MainPenalty = pLvlBasedPenalty
        # Then add an "escape clause" to prevent IM from exceeding mLvl
        MaximumPenalty = BilinearInterp(np.array([[0.,0.],[50.,50.]]),mLvlGrid,pLvlGrid)
        UninsuredPremiumFunc = LowerEnvelope2D(MainPenalty,MaximumPenalty)
        
        self.UninsuredPremiumFunc = UninsuredPremiumFunc
        
               
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
        ZeroFunc = ConstantFunction(0.)
        solution_now = InsuranceSelectionSolution(mLvlMin=ZeroFunc)
        
        HealthCount = 5
        for h in range(HealthCount):
            solution_now.appendSolution(vFunc=ZeroFunc,vPfunc=ZeroFunc,hLvl=ZeroFunc,AVfunc=[],
                                        policyFunc=[],vFuncByContract=[],
                                        CritDevFunc=[])
        
        self.solution_terminal = solution_now
                
        
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
        
        self.updateFirstOrderConditionFuncs()
        self.updateEffPriceList()
        
    
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
        # Make initial Markov state
        N = self.AgentCount
        base_draws = drawUniform(N,seed=self.RNG.randint(0,2**31-1))
        Cutoffs = np.cumsum(np.array(self.MrkvPrbsInit))
        MrkvNow = np.searchsorted(Cutoffs,base_draws).astype(int)
        
        # Make initial permanent income and asset levels, etc
        pLvlInit = np.exp(self.pLvlInitMean)*drawMeanOneLognormal(N=self.AgentCount,sigma=self.pLvlInitStd,seed=self.RNG.randint(0,2**31-1))
        self.aLvlInit = 0.01*pLvlInit
        pLvlNow = pLvlInit
        Live = np.ones(self.AgentCount,dtype=bool)
        Dead = np.logical_not(Live)
        PermShkNow = np.zeros_like(pLvlInit)
        TranShkNow = np.ones_like(pLvlInit)
        MedShkNow  = np.zeros_like(pLvlInit)
        DevNow  = np.zeros_like(pLvlInit)
        
        # Make blank histories for the state and shock variables
        T_sim = self.T_sim
        N_agents = N
        pLvlHist = np.zeros((T_sim,N_agents)) + np.nan
        MrkvHist = -np.ones((T_sim,N_agents),dtype=int)
        PrefShkHist = np.reshape(drawUniform(N=T_sim*N_agents,seed=self.RNG.randint(0,2**31-1)),(T_sim,N_agents))
        MedShkHist = np.zeros((T_sim,N_agents)) + np.nan
        DevHist = np.zeros((T_sim,N_agents)) + np.nan
        IncomeHist = np.zeros((T_sim,N_agents)) + np.nan
        IncAboveThreshHist = np.zeros((T_sim,N_agents))

        # Loop through each period of life and update histories
        t_cycle=0
        for t in range(T_sim):
            # Make sure pLvl doesn't go outside of grid boundaries
            pLvl_min = self.pLvlGrid[t_cycle][0]
            pLvl_max = self.pLvlGrid[t_cycle][-1]
            pLvlNow[Live] = np.minimum(np.maximum(pLvlNow[Live],pLvl_min),pLvl_max)
            
            # Add current states to histories
            pLvlHist[t,Live] = pLvlNow[Live]
            MrkvHist[t,Live] = MrkvNow[Live]
            IncomeHist[t,:] = pLvlNow*TranShkNow
            IncAboveThreshHist[t,Live] = np.maximum(IncomeHist[t,Live] - self.TaxFunc.threshold, 0.0)
            
            # Make sample MrkvArray to get now and next state counts
            ESImrkvFunc = self.ESImrkvFunc[t_cycle]
            ESImrkv_example = ESImrkvFunc(1.0)[0,:,:]
            ESIcountNow = ESImrkv_example.shape[0]
            ESIcountNext = ESImrkv_example.shape[1]
            
            # Get medical shocks for this period in each health state
            MedShkNow[:] = np.nan
            DevNow[:] = np.nan
            HealthNow = np.mod(MrkvNow,5)
            for h in range(5):
                these = HealthNow == h
                N = np.sum(these)
                
                sigma = self.MedShkStd[t_cycle][h]
                mu = self.MedShkAvg[t_cycle][h]
                DevDraws = drawNormal(N=N,seed=self.RNG.randint(0,2**31-1))
                
                zero_prb = self.ZeroMedShkPrb[t_cycle][h]
                zero_med = drawBernoulli(N=N,p=zero_prb,seed=self.RNG.randint(0,2**31-1))
                DevDraws[zero_med] = -np.inf
                MedShkDraws = np.exp(mu + sigma*DevDraws)
                DevNow[these] = DevDraws
                MedShkNow[these] = MedShkDraws
                
            # Determine which agents die based on their mortality probability
            LivPrb_temp = self.LivPrb[t_cycle][HealthNow[Live]]
            LivPrbAll = np.zeros_like(pLvlNow)
            LivPrbAll[Live] = LivPrb_temp
            MortShkNow = drawUniform(N=self.AgentCount,seed=self.RNG.randint(0,2**31-1))
            Dead = MortShkNow > LivPrbAll
            Live = np.logical_not(Dead)
            
            # Draw health states for next period
            HealthNext = copy(MrkvNow)
            for h in range(5):
                these = HealthNow == h
                events = np.arange(5)
                N = np.sum(these)
                probs = self.HealthMrkvArray[t_cycle][h,:]
                idx = drawDiscrete(N=N,P=probs,X=events,seed=self.RNG.randint(0,2**31-1)).astype(int)
                HealthNext[these] = idx
            
            # Draw ESI states for next period
            ESInow = MrkvNow / 5
            ESInext = copy(ESInow)
            ESImrkvArray_all = self.ESImrkvFunc[t_cycle](pLvlNow)
            for j in range(ESIcountNow):
                these = ESInow == j
                N = np.sum(these)
                
                ESIcumProbs = np.cumsum(ESImrkvArray_all[these,j,:],axis=1)
                ESIshkNow = drawUniform(N=N,seed=self.RNG.randint(0,2**31-1))
                ESIshkNow_tiled = np.tile(np.reshape(ESIshkNow,(N,1)),(1,ESIcountNext))
                ESInext[these] = np.sum(ESIshkNow_tiled > ESIcumProbs, axis=1)
            
            # Combine health and ESI states to get overall next Markov state
            MrkvNext = 5*ESInext + HealthNext
            MrkvNext[Dead] = -1 # Actually kill those who died
            MrkvNow = MrkvNext  # Next period will soon be this period

            # Get income shocks for next period in each health state
            PermShkNow[:] = np.nan
            TranShkNow[:] = np.nan
            for h in range(5):
                these = HealthNext == h
                N = np.sum(these)
                
                # First income shocks for next period...
                IncDstn_temp = self.IncomeDstn[t_cycle][h]
                probs = IncDstn_temp[0]
                events = np.arange(probs.size)
                idx = drawDiscrete(N=N,P=probs,X=events,seed=self.RNG.randint(0,2**31-1)).astype(int)
                PermShkNow[these] = IncDstn_temp[1][idx]
                TranShkNow[these] = IncDstn_temp[2][idx]
            
            # Store the medical shocks and update permanent income for next period
            MedShkHist[t,:] = MedShkNow
            DevHist[t,:] = DevNow
            pLvlNow = self.pLvlNextFunc[t](pLvlNow)*PermShkNow
            
            # Advance t_cycle or reset it to zero
            t_cycle += 1
            if t_cycle == self.T_cycle:
                t_cycle = 0
                
        # Make boolean arrays for health state for all agents
        HealthBoolArray = np.zeros((self.T_sim,self.AgentCount,5),dtype=bool)
        LiveBoolArray = MrkvHist >= 0
        for h in range(5):
            HealthBoolArray[:,:,h] = np.logical_and(np.mod(MrkvHist,5) == h, LiveBoolArray)
        self.HealthBoolArray = HealthBoolArray
        self.LiveBoolArray = LiveBoolArray
        
        # Make boolean array for overall Markov state for all agents
        MrkvBoolArray = np.zeros((self.T_sim,self.AgentCount,15),dtype=bool)
        for j in range(15):
            MrkvBoolArray[:,:,j] = MrkvHist == j
        self.MrkvBoolArray = MrkvBoolArray
            
        # Store the history arrays as attributes of self
        self.pLvlHist = pLvlHist
        self.IncomeHist = IncomeHist
        self.MrkvHist = MrkvHist
        self.MedShkHist = MedShkHist
        self.DevHist = DevHist
        self.PrefShkHist = PrefShkHist
        self.IncAboveThreshHist = IncAboveThreshHist
        self.IncAboveThreshByAge = np.sum(IncAboveThreshHist,axis=1)

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
        MrkvArrayCombined = combineIndepMrkvArrays(self.ESImrkvArray[t],self.HealthMrkvArray[t])
        StateCountNow = MrkvArrayCombined.shape[0]
        
        # Get state and shock vectors for (living) agents
        MrkvNow = self.MrkvHist[t,:]
        pLvlNow = self.pLvlHist[t,:]
        MedShkNow = self.MedShkHist[t,:]
        DevNow = self.DevHist[t,:]
        PrefShkNow = self.PrefShkHist[t,:]
        
        # Initialize simulation output arrays
        mLvlNow = np.zeros_like(self.aLvlNow) + np.nan
        aLvlNow = np.zeros_like(mLvlNow) + np.nan
        TaxNow = np.zeros_like(mLvlNow)
        cLvlNow = np.zeros_like(mLvlNow) + np.nan
        OOPnow = np.zeros_like(mLvlNow) + np.nan
        MedLvlNow = np.zeros_like(mLvlNow) + np.nan
        xLvlNow = np.zeros_like(mLvlNow) + np.nan
        PremNow = np.zeros_like(mLvlNow) + np.nan
        WelfareNow = np.zeros_like(mLvlNow)
        SubsidyNow = np.zeros_like(mLvlNow)
        MedicareNow = np.zeros_like(mLvlNow)
        ContractNow = -np.ones(self.AgentCount,dtype=int)
        InsProbNow = np.zeros_like(mLvlNow) + np.nan

        # Loop through each health state and get agents' controls
        for h in range(StateCountNow):
            h_alt = h % 5
            these = MrkvNow == h
            N = np.sum(these)
            Z = len(self.solution[t].policyFunc[h])
            
            # Get the pseudo-inverse value of holding each contract
            vNvrs_temp = np.zeros((N,Z)) + np.nan
            y_temp = self.IncomeHist[t,these]
            Dev_temp = DevNow[these]
            Shk_temp = MedShkNow[these]
            tax_temp = self.TaxFunc(y_temp)
            m_temp = self.aLvlNow[these]*self.Rfree[h_alt] + y_temp - tax_temp
            p_temp = pLvlNow[these]
            for z in range(Z):                
                Premium = self.solution[t].policyFunc[h][z].Contract.Premium(m_temp,y_temp)
                if (z > 0) and (h < 5):
                    Subsidy = np.minimum(self.SubsidyFunc[t][h](m_temp,y_temp,Premium), Premium)
                else:
                    Subsidy = np.zeros_like(Premium)
                vNvrs_temp[:,z] = self.solution[t].vFuncByContract[h][z].func(m_temp-(Premium-Subsidy),p_temp)
            
            if self.ChoiceShkMag[t] > 0.:
                # Get choice probabilities for each contract
                vNvrs_temp[np.isnan(vNvrs_temp)] = -np.inf
                v_best = np.max(vNvrs_temp,axis=1)
                v_best_big = np.tile(np.reshape(v_best,(N,1)),(1,Z))
                v_adj_exp = np.exp((vNvrs_temp - v_best_big)/self.ChoiceShkMag[t])
                v_sum_rep = np.tile(np.reshape(np.sum(v_adj_exp,axis=1),(N,1)),(1,Z))
                ChoicePrbs = v_adj_exp/v_sum_rep
                Cutoffs = np.cumsum(ChoicePrbs,axis=1)
                InsProbNow[these] = 1.0 - ChoicePrbs[:,0]
                
                # Select a contract for each agent based on the unified preference shock
                PrefShk_temp = np.tile(np.reshape(PrefShkNow[these],(N,1)),(1,Z))
                z_choice = np.sum(PrefShk_temp > Cutoffs,axis=1).astype(int)
            else:
                z_choice = np.argmax(vNvrs_temp,axis=1) # Just choose best one if there are no shocks
            
            # For each contract, get controls for agents who buy it
            c_temp = np.zeros(N) + np.nan
            a_temp = np.zeros(N) + np.nan
            Med_temp = np.zeros(N) + np.nan
            x_temp = np.zeros(N) + np.nan
            OOP_temp = np.zeros(N) + np.nan
            Prem_temp = np.zeros(N) + np.nan
            Welfare_temp = np.zeros(N) + np.nan
            Medicare_temp = np.zeros(N) + np.nan
            Subsidy_temp = np.zeros(N) + np.nan
            
            for z in range(Z):
                ThisPolicy = self.solution[t].policyFunc[h][z]
                ThisContract = ThisPolicy.Contract
                idx = z_choice == z
                Prem_here = ThisContract.Premium(m_temp[idx],p_temp[idx])
                if (z > 0) and (h < 5):
                    Subsidy_here = np.minimum(self.SubsidyFunc[t][h](m_temp[idx],y_temp[idx],Prem_here), Prem_here)
                else:
                    Subsidy_here = np.zeros_like(Prem_here)
                m_minus_prem = m_temp[idx] - (Prem_here - Subsidy_here)
                c_here,Med_here,x_here = ThisPolicy(m_minus_prem,p_temp[idx],Dev_temp[idx])
                CritDev_here = self.solution[t].CritDevFunc[h][z](m_minus_prem,p_temp[idx])
                EffPrice = ThisContract.Copay*MedPriceNow
                OOP_here = ThisContract.OOPfunc(Med_here)
                a_here = np.maximum(m_minus_prem - x_here, 0.0) # This shouldn't be necessary
                Welfare_here = np.zeros_like(a_here)
                
                # Handle the consumption floor
                NeedHelp = Dev_temp[idx] > CritDev_here
                a_here[NeedHelp]   = 0.
                c_here[NeedHelp]   = self.Cfloor
                Med_here[NeedHelp] = self.Cfloor**(self.CRRA/self.CRRAmed)*EffPrice**(-1./self.CRRAmed)*Shk_temp[idx][NeedHelp]**(1.-1/self.CRRAmed)
                OOP_here[NeedHelp] = np.maximum(m_minus_prem[NeedHelp] - self.Cfloor, 0.0)
                paid_by_insurance = MedPriceNow*Med_here - ThisContract.OOPfunc(Med_here)
                if z == 0:
                    Medicare_here = paid_by_insurance # If contract 0 paid out benefits, it's Medicare
                else:
                    Medicare_here = np.zeros_like(a_here)
                
                total_bundle_cost = self.Cfloor + MedPriceNow*Med_here[NeedHelp]
                Welfare_here[NeedHelp] = total_bundle_cost - m_minus_prem[NeedHelp] - paid_by_insurance[NeedHelp]
                
                # Store the values for this insurance contract in the temporary arrays for this Markov state
                c_temp[idx] = c_here
                a_temp[idx] = a_here
                Med_temp[idx] = np.maximum(Med_here,0.0) # Prevents numeric glitching
                x_temp[idx] = x_here
                OOP_temp[idx] = OOP_here
                Prem_temp[idx] = Prem_here
                Welfare_temp[idx] = Welfare_here
                Medicare_temp[idx] = Medicare_here
                Subsidy_temp[idx] = Subsidy_here
            
            # Store the controls for this health
            mLvlNow[these] = m_temp
            aLvlNow[these] = a_temp
            TaxNow[these] = tax_temp
            cLvlNow[these] = c_temp
            MedLvlNow[these] = Med_temp
            xLvlNow[these] = x_temp
            OOPnow[these] = OOP_temp
            PremNow[these] = Prem_temp
            WelfareNow[these] = Welfare_temp
            MedicareNow[these] = Medicare_temp
            SubsidyNow[these] = Subsidy_temp
            ContractNow[these] = z_choice
        
        # Calculate revenues from individual mandate:
        # Premiums paid by uninsured people who aren't offered ESI are actually mandate taxes
        NotOffered = np.logical_and(MrkvNow >=0, MrkvNow < 5)
        Uninsured = ContractNow == 0
        SubjectToIM = np.logical_and(NotOffered,Uninsured)
        IMpenaltyNow = np.zeros_like(mLvlNow)
        IMpenaltyNow[SubjectToIM] = PremNow[SubjectToIM] # 
        
        # Calculate total spending
        SpendNow = WelfareNow + SubsidyNow + MedicareNow

        # Calculate end of period assets and store results as attributes of self
        self.mLvlNow = mLvlNow
        self.BudgetNow = SpendNow - IMpenaltyNow
        self.aLvlNow = aLvlNow
        self.cLvlNow = cLvlNow
        self.MedLvlNow = MedLvlNow
        self.OOPnow = OOPnow
        self.PremNow = PremNow
        self.ContractNow = ContractNow
        self.InsProbNow = InsProbNow
        
        
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
        MaxContracts = 0
        MaxStateCount = 0
        T_working = 40
        for t in range(T_working):
            StateCount = len(self.ContractList[t])
            MaxContracts = np.maximum(MaxContracts,max([len(self.ContractList[t][h]) for h in range(StateCount)]))
            MaxStateCount = np.maximum(MaxStateCount,StateCount)
        
        ExpInsPay = np.zeros((T_working,MaxStateCount,MaxContracts))
        ExpBuyers = np.zeros((T_working,MaxStateCount,MaxContracts))
        
        for t in range(T_working):
            StateCount = len(self.ContractList[t])
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
                    if z > 0:
                        if j > 10:
                            Premium = np.maximum(self.PremiumFuncs[t][j][z](mLvl,pLvl) - self.EmpContr, 0.0) # Premium net of employer contribution
                        elif j < 5:
                            Premium = self.PremiumFuncs[t][j][z](mLvl,pLvl)
                            Subsidy = self.SubsidyFunc[t][j](mLvl,pLvl,Premium)
                            Premium = np.maximum(Premium - Subsidy, 0.0) # Premium net of IMI subsidy
                        else:
                            Premium = self.PremiumFuncs[t][j][z](mLvl,pLvl) # Full premium
                    else:
                        if (j < 5) or self.MandateForESI:
                            Premium = self.UninsuredPremiumFunc(mLvl,pLvl) # Uninsured, IMI (or ESI with mandate applicable)
                        else:
                            Premium = np.zeros_like(mLvl)
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
    
    
    def plotvFunc(self,t,p,mMin=0.0,mMax=10.0,H=None,decurve=True,savename=None):
        mLvl = np.linspace(mMin,mMax,200)
        if H is None:
            H = range(len(self.solution[t].vFunc))
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
        if savename is not None:
            plt.savefig('../Figures/' + savename + '.pdf')
        plt.show()
        
    def plotvPfunc(self,t,p,mMin=0.0,mMax=10.0,H=None,decurve=True,savename=None):
        mLvl = np.linspace(mMin,mMax,200)
        if H is None:
            H = range(len(self.solution[t].vFunc))
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
        if savename is not None:
            plt.savefig('../Figures/' + savename + '.pdf')
        plt.show()
        
    def plotvFuncByContract(self,t,h,p,mMin=0.0,mMax=10.0,Z=None,savename=None):
        print('Pseudo-inverse value function by contract:')
        mLvl = np.linspace(mMin,mMax,200)
        if Z is None:
            Z = range(len(self.solution[t].vFuncByContract[h]))
        for z in Z:
            f = lambda x : self.solution[t].vFuncByContract[h][z].func(x,p*np.ones_like(x))
            Prem = self.ContractList[t][h][z].Premium(0.,0.) # Need to fix this if want to see IM in action
            plt.plot(mLvl+Prem,f(mLvl))
        plt.xlabel('Market resources mLvl')
        plt.ylabel('Contract pseudo-inverse value uinv(v(mLvl))')
        plt.ylim(ymin=0.0)
        if savename is not None:
            plt.savefig('../Figures/' + savename + '.pdf')
        plt.show()
        
    def plotcFuncByContract(self,t,h,p,MedShk,mMin=0.0,mMax=10.0,Z=None):
        mLvl = np.linspace(mMin,mMax,200)
        if Z is None:
            Z = range(len(self.solution[t].policyFunc[h]))
        for z in Z:
            cLvl,MedLvl,xLvl = self.solution[t].policyFunc[h][z](mLvl,p*np.ones_like(mLvl),MedShk*np.ones_like(mLvl))
            plt.plot(mLvl,cLvl)
        plt.xlabel('Market resources mLvl')
        plt.ylabel('Consumption c(mLvl)')
        plt.ylim(ymin=0.0)
        plt.show()
        
    def plotMedFuncByContract(self,t,h,p,MedShk,mMin=0.0,mMax=10.0,Z=None,savename=None):
        mLvl = np.linspace(mMin,mMax,200)
        if Z is None:
            Z = range(len(self.solution[t].policyFunc[h]))
        for z in Z:
            cLvl,MedLvl = self.solution[t].policyFunc[h][z](mLvl,p*np.ones_like(mLvl),MedShk*np.ones_like(mLvl))
            plt.plot(mLvl,MedLvl)
        plt.xlabel('Market resources mLvl')
        plt.ylabel('Medical care Med(mLvl)')
        plt.ylim(ymin=0.0)
        if savename is not None:
            plt.savefig('../Figures/' + savename + '.pdf')
        plt.show()
        
    def plotcFuncByDev(self,t,h,z,p,mMin=0.0,mMax=10.0,DevSet=None,savename=None):        
        mLvl = np.linspace(mMin,mMax,200)
        if DevSet is None:
            DevSet = np.linspace(self.DevMin,self.DevMax,self.MedShkCount)
        for Dev in DevSet:            
            cLvl = self.solution[t].policyFunc[h][z].cFunc(mLvl,p*np.ones_like(mLvl),Dev*np.ones_like(mLvl))
            plt.plot(mLvl,cLvl)
        plt.xlabel('Market resources mLvl')
        plt.ylabel('Consumption c(mLvl)')
        plt.ylim(ymin=0.0)
        if savename is not None:
            plt.savefig('../Figures/' + savename + '.pdf')
        plt.show()
        
    def plotMedFuncByDev(self,t,h,z,p,mMin=0.0,mMax=10.0,DevSet=None,savename=None):        
        mLvl = np.linspace(mMin,mMax,200)
        if DevSet is None:
            DevSet = np.linspace(self.DevMin,self.DevMax,self.MedShkCount)
        for Dev in DevSet:            
            MedLvl = self.solution[t].policyFunc[h][z].MedFunc(mLvl,p*np.ones_like(mLvl),Dev*np.ones_like(mLvl))
            plt.plot(mLvl,MedLvl)
        plt.xlabel('Market resources mLvl')
        plt.ylabel('Medical care Med(mLvl)')
        plt.ylim(ymin=0.0)
        if savename is not None:
            plt.savefig('../Figures/' + savename + '.pdf')
        plt.show()
        
    def plotxFuncByDev(self,t,h,z,p,mMin=0.0,mMax=10.0,DevSet=None,savename=None):        
        mLvl = np.linspace(mMin,mMax,200)
        if DevSet is None:
            DevSet = np.linspace(self.DevMin,self.DevMax,self.MedShkCount)
        for Dev in DevSet:            
            xLvl = self.solution[t].policyFunc[h][z].xFunc(mLvl,p*np.ones_like(mLvl),Dev*np.ones_like(mLvl))
            plt.plot(mLvl,xLvl)
        plt.xlabel('Market resources mLvl')
        plt.ylabel('Expenditure x(mLvl)')
        plt.ylim(ymin=0.0)
        if savename is not None:
            plt.savefig('../Figures/' + savename + '.pdf')
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
