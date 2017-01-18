'''
This is a first attempt at the DynInsSel estimation.  It has functions to construct agent type(s)
from input parameters; it will eventually have the ability to compare simulated data to real data.
'''
import sys 
sys.path.insert(0,'../../')

import numpy as np
import DynInsSelParameters as Params
from copy import copy
from InsuranceSelectionModel import MedInsuranceContract, InsSelConsumerType
from HARKinterpolation import ConstantFunction
from HARKutilities import approxUniform

def makeDynInsSelType(CRRAcon,CRRAmed,DiscFac,ChoiceShkMag,MedShkMeanAgeParams,MedShkMeanVGparams,
                      MedShkMeanGDparams,MedShkMeanFRparams,MedShkMeanPRparams,MedShkStdAgeParams,
                      MedShkStdVGparams,MedShkStdGDparams,MedShkStdFRparams,MedShkStdPRparams,
                      PremiumArray,PremiumSubsidy,EducType,InsChoiceBool):
    '''
    Makes an InsSelConsumerType using (human-organized) structural parameters for the estimation.
    
    Parameters
    ----------
    CRRAcon : float
        Coefficient of relative risk aversion for consumption.
    CRRAmed : float
        Coefficient of relative risk aversion for medical care.
    DiscFac : float
        Intertemporal discount factor.
    ChoiceShkMag : float
        Standard deviation of preference shocks over insurance contracts.
    MedShkMeanAgeParams : [float]
        Quartic parameters for the mean of log medical need shocks by age, for excellent health.
    MedShkMeanVGparams: [float]
        Linear parameters for the mean of log medical need shocks by age: adjuster for very good health.
    MedShkMeanGDparams: [float]
        Linear parameters for the mean of log medical need shocks by age: adjuster for good health.
    MedShkMeanGDparams: [float]
        Linear parameters for the mean of log medical need shocks by age: adjuster for fair health.
    MedShkMeanGDparams: [float]
        Linear parameters for the mean of log medical need shocks by age: adjuster for poor health.
    MedShkStdAgeParams : [float]
        Quartic parameters for the stdev of log medical need shocks by age, for excellent health.
    MedShkStdVGparams: [float]
        Linear parameters for the stdev of log medical need shocks by age: adjuster for very good health.
    MedShkStdGDparams: [float]
        Linear parameters for the stdev of log medical need shocks by age: adjuster for good health.
    MedShkStdGDparams: [float]
        Linear parameters for the stdev of log medical need shocks by age: adjuster for fair health.
    MedShkStdGDparams: [float]
        Linear parameters for the stdev of log medical need shocks by age: adjuster for poor health.
    PremiumArray : np.array
        Array of constant premiums for the N insurance contracts.
    PremiumSubsidy : float
        Employer contribution to any (non-null) insurance contract.
    EducType : int
        Discrete education type.  0 --> dropout, 1 --> high school, 2 --> college graduate.
    InsChoiceBool : boolean
        Indicator for whether insurance choice is in the model.  When True, working-age agents can
        choose among several insurance contracts, while retirees choose between basic Medicare,
        Medicare A+B, or Medicare A+B + Medigap.  When false, agents get exogenous contracts.
    
    Returns
    -------
    ThisType : InsSelConsumerType
        The constructed agent type.
    '''
                          
    # Make a dictionary from the parameter file depending on education (already has most parameters)
    if EducType == 0:
        TypeDict = copy(Params.DropoutDictionary)
    elif EducType == 1:
        TypeDict = copy(Params.HighschoolDictionary)
    elif EducType == 2:
        TypeDict = copy(Params.CollegeDictionary)
    else:
        assert False, 'EducType must be 0, 1, or 2!'
    TypeDict['CRRA'] = CRRAcon
    TypeDict['CRRAmed'] = CRRAmed
    TypeDict['DiscFac'] = DiscFac
    TypeDict['ChoiceShkMag'] = Params.AgeCount*[ChoiceShkMag]
                          
    # Make arrays of medical shock means and standard deviations
    MedShkMeanArray = np.zeros((5,Params.AgeCount)) + np.nan
    MedShkStdArray  = np.zeros((5,Params.AgeCount)) + np.nan
    AgeArray = np.arange(Params.AgeCount,dtype=float)
    MedShkMeanEXfunc = lambda age : MedShkMeanAgeParams[0] + MedShkMeanAgeParams[1]*age + MedShkMeanAgeParams[2]*age**2 \
                        + MedShkMeanAgeParams[3]*age**3 + MedShkMeanAgeParams[4]*age**4
    MedShkMeanVGfunc = lambda age : MedShkMeanEXfunc(age) + MedShkMeanVGparams[0] + MedShkMeanVGparams[1]*age
    MedShkMeanGDfunc = lambda age : MedShkMeanVGfunc(age) + MedShkMeanGDparams[0] + MedShkMeanGDparams[1]*age
    MedShkMeanFRfunc = lambda age : MedShkMeanGDfunc(age) + MedShkMeanFRparams[0] + MedShkMeanFRparams[1]*age
    MedShkMeanPRfunc = lambda age : MedShkMeanFRfunc(age) + MedShkMeanPRparams[0] + MedShkMeanPRparams[1]*age
    MedShkStdEXfunc = lambda age : MedShkStdAgeParams[0] + MedShkStdAgeParams[1]*age + MedShkStdAgeParams[2]*age**2 \
                        + MedShkStdAgeParams[3]*age**3 + MedShkStdAgeParams[4]*age**4
    MedShkStdVGfunc = lambda age : MedShkStdEXfunc(age) + MedShkStdVGparams[0] + MedShkStdVGparams[1]*age
    MedShkStdGDfunc = lambda age : MedShkStdVGfunc(age) + MedShkStdGDparams[0] + MedShkStdGDparams[1]*age
    MedShkStdFRfunc = lambda age : MedShkStdGDfunc(age) + MedShkStdFRparams[0] + MedShkStdFRparams[1]*age
    MedShkStdPRfunc = lambda age : MedShkStdFRfunc(age) + MedShkStdPRparams[0] + MedShkStdPRparams[1]*age
    MedShkMeanArray[4,:] = np.exp(MedShkMeanEXfunc(AgeArray))
    MedShkMeanArray[3,:] = np.exp(MedShkMeanVGfunc(AgeArray))
    MedShkMeanArray[2,:] = np.exp(MedShkMeanGDfunc(AgeArray))
    MedShkMeanArray[1,:] = np.exp(MedShkMeanFRfunc(AgeArray))
    MedShkMeanArray[0,:] = np.exp(MedShkMeanPRfunc(AgeArray))
    MedShkStdArray[4,:] = np.exp(MedShkStdEXfunc(AgeArray))
    MedShkStdArray[3,:] = np.exp(MedShkStdVGfunc(AgeArray))
    MedShkStdArray[2,:] = np.exp(MedShkStdGDfunc(AgeArray))
    MedShkStdArray[1,:] = np.exp(MedShkStdFRfunc(AgeArray))
    MedShkStdArray[0,:] = np.exp(MedShkStdPRfunc(AgeArray))
    TypeDict['MedShkAvg'] = MedShkMeanArray.transpose().tolist()
    TypeDict['MedShkStd'] = MedShkStdArray.transpose().tolist()
    
    # Make insurance contracts when working and retired, then combine into a lifecycle list
    WorkingContractList = []
    if InsChoiceBool:
        WorkingContractList.append(MedInsuranceContract(ConstantFunction(0.0),0.0,1.0,Params.MedPrice))
        for j in range(PremiumArray.size):
            Premium = max([PremiumArray[j] - PremiumSubsidy,0.0])
            Copay = 0.05
            Deductible = Params.DeductibleList[j]
            WorkingContractList.append(MedInsuranceContract(ConstantFunction(Premium),Deductible,Copay,Params.MedPrice))
    else:
        WorkingContractList.append(MedInsuranceContract(ConstantFunction(0.0),0.2,0.05,Params.MedPrice))
    RetiredContractList = [MedInsuranceContract(ConstantFunction(0.0),0.0,0.1,Params.MedPrice)]
    TypeDict['ContractList'] = Params.working_T*[5*[WorkingContractList]] + Params.retired_T*[5*[RetiredContractList]]
    
    # Make and return a DynInsSelType
    ThisType = InsSelConsumerType(**TypeDict)
    return ThisType
        

def makeAllTypesFromParams(ParamArray,PremiumArray,InsChoiceBool):
    '''
    Makes a list of 3 or 24 InsSelConsumerTypes, to be used for estimation.
    
    Parameters
    ----------
    ParamArray : np.array
        Array of size 33, representing all of the structural parameters.
    PremiumArray : np.array
        Array of premiums for insurance contracts for workers. Irrelevant if InsChoiceBool = False.
    InsChoiceBool : boolean
        Indicator for whether the agents should have a choice of insurance contract.
        
    Returns
    -------
    AgentList : [InsSelConsumerType]
        List of InsSelConsumerTypes to be used in estimation.
    '''
    # Unpack the parameters
    DiscFac = ParamArray[0]
    CRRAcon = ParamArray[1]
    CRRAmed = (1.0 + np.exp(ParamArray[2]))*CRRAcon
    ChoiceShkMag = np.exp(ParamArray[3])
    SubsidyZeroRate = 1.0/(1.0 + np.exp(ParamArray[4]))
    SubsidyAvg = np.exp(ParamArray[5])
    SubsidyWidth = SubsidyAvg/(1.0 + np.exp(ParamArray[6]))
    MedShkMeanAgeParams = ParamArray[7:12]
    MedShkMeanVGparams = ParamArray[12:14]
    MedShkMeanGDparams = ParamArray[14:16]
    MedShkMeanFRparams = ParamArray[16:18]
    MedShkMeanPRparams = ParamArray[18:20]
    MedShkStdAgeParams = ParamArray[20:25]
    MedShkStdVGparams = ParamArray[25:27]
    MedShkStdGDparams = ParamArray[27:29]
    MedShkStdFRparams = ParamArray[29:31]
    MedShkStdPRparams = ParamArray[31:33]
    
    # Make the array of premium subsidies (trivial if there is no insurance choice)
    if InsChoiceBool:
        Temp = approxUniform(7,SubsidyAvg-SubsidyWidth,SubsidyAvg+SubsidyWidth)
        SubsidyArray = Temp[1]
        SubsidyArray = np.insert(SubsidyArray,0,0.0)
        WeightArray  = np.insert(Temp[0]*(1.0-SubsidyZeroRate),0,SubsidyZeroRate)
    else:
        SubsidyArray = np.array([0.0])
        WeightArray  = np.array([1.0])
    
    # Make the list of types
    AgentList = []
    for j in range(SubsidyArray.size):
        for k in range(3):
            AgentList.append(makeDynInsSelType(CRRAcon,CRRAmed,DiscFac,ChoiceShkMag,MedShkMeanAgeParams,
                      MedShkMeanVGparams,MedShkMeanGDparams,MedShkMeanFRparams,MedShkMeanPRparams,
                      MedShkStdAgeParams,MedShkStdVGparams,MedShkStdGDparams,MedShkStdFRparams,
                      MedShkStdPRparams,PremiumArray,SubsidyArray[j],k,InsChoiceBool))
            AgentList[-1].Weight = WeightArray[j]*Params.EducWeight[k]
    return AgentList
    
    
    
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    
    InsChoice = False
    AgentList = makeAllTypesFromParams(Params.test_param_vec,np.array([1,2,3,4,5]),InsChoice)
    MyType = AgentList[0]
    MyType.preSolve()
    
    print('Pseudo-inverse value function by contract:')
    mLvl = np.linspace(0,5,200)
    h = 1
    J = len(MyType.solution_terminal.vFuncByContract[h])
    for j in range(J):
        f = lambda x : MyType.solution_terminal.vFuncByContract[h][j].func(x,np.ones_like(x))
        Prem = MyType.ContractList[-1][h][j].Premium(0)
        plt.plot(mLvl+Prem,f(mLvl))
    f = lambda x : MyType.solution_terminal.vFunc[h].func(x,np.ones_like(x))
    plt.plot(mLvl,f(mLvl))
    plt.show()    
    
    print('Pseudo-inverse value function by health:')
    mLvl = np.linspace(0,5,200)
    H = len(MyType.LivPrb[0])
    for h in range(H):
        f = lambda x : MyType.solution_terminal.vFunc[h].func(x,np.ones_like(x))
        plt.plot(mLvl,f(mLvl))
    plt.show()
    
    print('Pseudo-inverse marginal value function by health:')
    mLvl = np.linspace(0,0.25,200)
    H = len(MyType.LivPrb[0])
    for h in range(H):
        f = lambda x : MyType.solution_terminal.vPfunc[h].cFunc(x,np.ones_like(x))
        plt.plot(mLvl,f(mLvl))
    plt.show()
    
    h = 1
    print('Consumption function by contract:')
    mLvl = np.linspace(0,5,200)
    J = len(MyType.solution_terminal.vFuncByContract[h])
    for j in range(J):
        cLvl,MedLvl = MyType.solution_terminal.policyFunc[h][j](mLvl,1*np.ones_like(mLvl),1*np.ones_like(mLvl))
        plt.plot(mLvl,cLvl)
    plt.show()
    
    print('Pseudo-inverse value function by coinsurance rate:')
    mLvl = np.linspace(0,5,200)
    J = len(MyType.solution_terminal.vFuncByContract[0])
    for j in range(J):
        v = MyType.solution_terminal.policyFunc[0][j].ValueFuncCopay.vFuncNvrs(mLvl,np.ones_like(mLvl),1*np.ones_like(mLvl))
        plt.plot(mLvl,v)
    plt.show()
    