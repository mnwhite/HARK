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
from HARKutilities import approxUniform, getPercentiles
from HARKcore import Market


class DynInsSelType(InsSelConsumerType):
    '''
    An extension of InsSelConsumerType that adds and adjusts methods for estimation.
    '''    
    def makeBooleanArrays(self):
        '''
        Makes the attribute IncQuintBoolArray, specifying which income quintile
        each agent is in for each period of life.  Should only be run after the
        economy runs the method getIncomeQuintiles().  Also makes the attribute
        HealthBoolArray, indicating health by age.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        None
        '''
        IncQuintBoolArray = np.zeros((self.T_cycle,self.AgentCount,5),dtype=bool)
        for t in range(self.T_cycle):
            for q in range(5):
                bot = self.IncomeQuintiles[t,q-1]
                top = self.IncomeQuintiles[t,q]
                if q == 0:
                    bot = 0.0
                if q == 4:
                    top = np.inf
                IncQuintBoolArray[t,:,q] = np.logical_and(self.pLvlHist[t,:] >= bot, self.pLvlHist[t,:] < top)
        self.IncQuintBoolArray = IncQuintBoolArray
        
        # Make boolean arrays for health state for all agents
        HealthBoolArray = np.zeros((self.T_cycle,self.AgentCount,5))
        for h in range(5):
            HealthBoolArray[:,:,h] = self.MrkvHist == h
        self.HealthBoolArray = HealthBoolArray
        self.LiveBoolArray = np.any(HealthBoolArray,axis=2)
        
    def preSolve(self):
        self.updateSolutionTerminal()
        
    def initializeSim(self):
        InsSelConsumerType.initializeSim(self)
        if 'ContractNow' in self.track_vars:
            self.ContractNow_hist = np.zeros_like(self.ContractNow_hist).astype(int)
            
    def postSim(self):
        MedPrice_temp = np.tile(np.reshape(self.MedPrice[0:self.T_sim],(self.T_sim,1)),(1,self.AgentCount))
        self.TotalMedHist = self.MedLvlNow_hist*MedPrice_temp
        self.WealthRatioHist = self.aLvlNow_hist/self.pLvlNowHist
        self.InsuredBoolArray = self.ContractNow_hist > 0
    
class DynInsSelMarket(Market):
    '''
    A class for representing the "insurance economy" with many agent types.
    '''    
    def calcSimulatedMoments(self):
        '''
        Calculates all simulated moments for this economy's AgentTypes.
        Should only be run after all types have solved and simulated.
        Stores results in attributes of self, to be combined into a single
        simulated moment array by combineSimMoments().
        
        Parameters
        ----------
        None
            
        Returns
        -------
        None
        '''
        Agents = self.Agents
        
        # Initialize moments by age
        WealthMedianByAge = np.zeros(40) + np.nan
        LogTotalMedMeanByAge = np.zeros(60) + np.nan
        LogTotalMedStdByAge = np.zeros(60) + np.nan
        InsuredRateByAge = np.zeros(40) + np.nan
        ZeroSubsidyRateByAge = np.zeros(40) + np.nan
        PremiumMeanByAge = np.zeros(40) + np.nan
        PremiumStdByAge = np.zeros(40) + np.nan

        # Calculate all simulated moments by age
        for t in range(60):
            WealthRatioList = []
            LogTotalMedList = []
            PremiumList = []
            LiveCount = 0.0
            InsuredCount = 0.0
            ZeroCount = 0.0
            for ThisType in Agents:
                these = ThisType.LiveBoolArray[t,:]
                LogTotalMedList.append(np.log(ThisType.TotalMedHist[t,these]+0.0001))
                if t < 40:
                    WealthRatioList.append(ThisType.WealthRatioHist[t,these])
                    these = ThisType.InsuredBoolArray[t,:]
                    PremiumList.append(ThisType.Premium_hist[t,these])
                    LiveCount += np.sum(ThisType.LiveBoolArray[t,:])
                    temp = np.sum(ThisType.InsuredBoolArray[t,:])
                    InsuredCount += temp
                    if ThisType.ZeroSubsidyool:
                        ZeroCount += temp
            if t < 40:
                WealthRatioArray = np.hcat(WealthRatioList)
                WealthMedianByAge[t] = np.median(WealthRatioArray)
                PremiumArray = np.hcat(PremiumList)
                PremiumMeanByAge[t] = np.mean(PremiumArray)
                PremiumStdByAge[t] = np.std(PremiumArray)
                InsuredRateByAge[t] = InsuredCount/LiveCount
                ZeroSubsidyRateByAge[t] = ZeroCount/InsuredCount
            LogTotalMedArray = np.hcat(LogTotalMedList)
            LogTotalMedMeanByAge[t] = np.mean(LogTotalMedArray)
            LogTotalMedStdByAge[t] = np.std(LogTotalMedArray)
            
        # Initialize moments by age-health and define age band bounds
        LogTotalMedMeanByAgeHealth = np.zeros((12,5)) + np.nan
        LogTotalMedStdByAgeHealth = np.zeros((12,5)) + np.nan
        AgeBounds = [[0,5],[5,10],[10,15],[15,20],[20,25],[25,30],[30,35],[35,40],[40,45],[45,50],[50,55],[55,60]]

        # Calculate all simulated moments by age-health
        for a in range(12):
            bot = AgeBounds[a,0]
            top = AgeBounds[a,1]
            for h in range(5):
                LogTotalMedList = []
                for ThisType in Agents:
                    these = ThisType.HealthBoolArray[bot:top,:,h]
                    LogTotalMedList.append(np.log(ThisType.TotalMedHist[bot:top,:][these]+0.0001))
                LogTotalMedArray = np.hcat(LogTotalMedList)
                LogTotalMedMeanByAgeHealth[a,h] = np.mean(LogTotalMedArray)
                LogTotalMedStdByAgeHealth[a,h] = np.std(LogTotalMedArray)
                
        # Initialize moments by age-income
        WealthMedianByAgeIncome = np.zeros((8,5)) + np.nan
        LogTotalMedMeanByAgeIncome = np.zeros((8,5)) + np.nan
        LogTotalMedStdByAgeIncome = np.zeros((8,5)) + np.nan
        InsuredRateByAgeIncome = np.zeros((8,5)) + np.nan
        PremiumMeanByAgeIncome = np.zeros((8,5)) + np.nan

        # Calculated all simulated moments by age-income
        for a in range(8):
            bot = AgeBounds[a,0]
            top = AgeBounds[a,1]
            for i in range(5):
                WealthRatioList = []
                LogTotalMedList = []
                PremiumList = []
                LiveCount = 0.0
                InsuredCount = 0.0
                for ThisType in Agents:
                    these = ThisType.IncQuintBoolArray[bot:top,:,h]
                    LiveCount += np.sum(ThisType.LiveBoolArray[bot:top,:][these])
                    LogTotalMedList.append(np.log(ThisType.TotalMedHist[bot:top,:][these]+0.0001))
                    WealthRatioList.append(ThisType.WealthRatioHist[bot:top,:][these])
                    these = np.logical_and(ThisType.InsuredBoolArray[bot:top,:],these)
                    PremiumList.append(ThisType.Premium_hist[bot:top,:][these])                    
                    temp = np.sum(ThisType.InsuredBoolArray[bot:top,:][these])
                    InsuredCount += temp
                WealthRatioArray = np.hcat(WealthRatioList)
                WealthMedianByAgeIncome[a,h] = np.median(WealthRatioArray)
                PremiumArray = np.hcat(PremiumList)
                PremiumMeanByAgeIncome[a,h] = np.mean(PremiumArray)
                InsuredRateByAgeIncome[a,h] = InsuredCount/LiveCount
                LogTotalMedArray = np.hcat(LogTotalMedList)
                LogTotalMedMeanByAgeIncome[a,h] = np.mean(LogTotalMedArray)
                LogTotalMedStdByAgeIncome[a,h] = np.std(LogTotalMedArray)
        
        # Store all of the simulated moments as attributes of self
        self.WealthMedianByAge = WealthMedianByAge
        self.LogTotalMedMeanByAge = LogTotalMedMeanByAge
        self.LogTotalMedStdByAge = LogTotalMedStdByAge
        self.InsuredRateByAge = InsuredRateByAge
        self.ZeroSubsidyRateByAge = ZeroSubsidyRateByAge
        self.PremiumMeanByAge = PremiumMeanByAge
        self.PremiumStdByAge = PremiumStdByAge
        self.LogTotalMedMeanByAgeHealth = LogTotalMedMeanByAgeHealth
        self.LogTotalMedStdByAgeHealth = LogTotalMedStdByAgeHealth
        self.WealthMedianByAgeIncome = WealthMedianByAgeIncome
        self.LogTotalMedMeanByAgeIncome = LogTotalMedMeanByAgeIncome
        self.LogTotalMedStdByAgeIncome = LogTotalMedStdByAgeIncome
        self.InsuredRateByAgeIncome = InsuredRateByAgeIncome
        self.PremiumMeanByAgeIncome = PremiumMeanByAgeIncome
        
    def getIncomeQuintiles(self):
        '''
        Calculates permanent income quintile cutoffs by age, across all AgentTypes.
        Result is stored as attribute IncomeQuintiles in each AgentType.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        None
        '''
        Cuts = np.array([0.2,0.4,0.6,0.8])
        IncomeQuintiles = np.zeros((self.Agents[0].T_cycle,Cuts.size))
        
        # Get income quintile cut points for each age
        for t in range(self.Agents[0].T_cycle):
            pLvlList = []
            for ThisType in self.Agents:
                pLvlList.append(ThisType.pLvlHist[t,:][ThisType.LivBoolArray[t,:]])
            pLvlArray = np.hcat(pLvlList)
            IncomeQuintiles[t,:] = getPercentiles(pLvlArray,percentiles = Cuts)
        
        # Store the income quintile cut points in each AgentType
        for ThisType in self.Agents:
            ThisType.IncomeQuintiles = IncomeQuintiles
                    
               
                
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
    ThisType : DynInsSelType
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
    ThisType = DynInsSelType(**TypeDict)
    ThisType.track_vars = ['aLvlNow','cLvlNow','MedLvlNow','PremNow','ContractNow']
    return ThisType
        

def makeAllTypesFromParams(ParamArray,PremiumArray,InsChoiceBool):
    '''
    Makes a list of 3 or 24 DynInsSelTypes, to be used for estimation.
    
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
    AgentList : [DynInsSelType]
        List of DynInsSelTypes to be used in estimation.
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
    from time import clock
    mystr = lambda number : "{:.4f}".format(number)
    
    InsChoice = False
    AgentList = makeAllTypesFromParams(Params.test_param_vec,np.array([1,2,3,4,5]),InsChoice)
    MyType = AgentList[1]

    t_start = clock()
    MyType.update()
    MyType.solve()
    MyType.makeShockHistory()
    MyType.initializeSim
    t_end = clock()
    print('Solving the agent took ' + mystr(t_end-t_start) + ' seconds.')
    
#    print('Terminal pseudo-inverse value function by contract:')
#    mLvl = np.linspace(0,5,200)
#    h = 1
#    J = len(MyType.solution_terminal.vFuncByContract[h])
#    for j in range(J):
#        f = lambda x : MyType.solution_terminal.vFuncByContract[h][j].func(x,np.ones_like(x))
#        Prem = MyType.ContractList[-1][h][j].Premium(0)
#        plt.plot(mLvl+Prem,f(mLvl))
#    f = lambda x : MyType.solution_terminal.vFunc[h].func(x,np.ones_like(x))
#    plt.plot(mLvl,f(mLvl))
#    plt.show()    
#    
#    print('Terminal pseudo-inverse value function by health:')
#    mLvl = np.linspace(0,5,200)
#    H = len(MyType.LivPrb[0])
#    for h in range(H):
#        f = lambda x : MyType.solution_terminal.vFunc[h].func(x,np.ones_like(x))
#        plt.plot(mLvl,f(mLvl))
#    plt.show()
#    
#    print('Terminal pseudo-inverse marginal value function by health:')
#    mLvl = np.linspace(0,0.25,200)
#    H = len(MyType.LivPrb[0])
#    for h in range(H):
#        f = lambda x : MyType.solution_terminal.vPfunc[h].cFunc(x,np.ones_like(x))
#        plt.plot(mLvl,f(mLvl))
#    plt.show()
#    
#    h = 1
#    print('Terminal pseudo-inverse value function by coinsurance rate:')
#    mLvl = np.linspace(0,5,200)
#    J = len(MyType.solution_terminal.vFuncByContract[0])
#    for j in range(J):
#        v = MyType.solution_terminal.policyFunc[0][j].ValueFuncCopay.vFuncNvrs(mLvl,np.ones_like(mLvl),1*np.ones_like(mLvl))
#        plt.plot(mLvl,v)
#    plt.show()
#    
#    print('Terminal consumption function by contract:')
#    mLvl = np.linspace(0,5,200)
#    J = len(MyType.solution_terminal.vFuncByContract[h])
#    for j in range(J):
#        cLvl,MedLvl = MyType.solution_terminal.policyFunc[h][j](mLvl,1*np.ones_like(mLvl),1*np.ones_like(mLvl))
#        plt.plot(mLvl,cLvl)
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
    
    h = 4    
    
    print('Pseudo-inverse value function by contract:')
    mLvl = np.linspace(0,10,200)
    J = len(MyType.solution[0].vFuncByContract[h])
    for j in range(J):
        f = lambda x : MyType.solution[0].vFuncByContract[h][j].func(x,p*np.ones_like(x))
        Prem = MyType.ContractList[0][h][j].Premium(0)
        plt.plot(mLvl+Prem,f(mLvl))
    f = lambda x : MyType.solution[0].vFunc[h].func(x,p*np.ones_like(x))
    plt.plot(mLvl,f(mLvl),'-k')
    plt.show()
    
    MedShk = 1.0e-2
    
    print('Pseudo-inverse value function by coinsurance rate:')
    mLvl = np.linspace(0,10,200)
    J = len(MyType.solution[0].vFuncByContract[h])
    v = MyType.solution[0].policyFunc[h][0].ValueFuncFullPrice.vFuncNvrs(mLvl,p*np.ones_like(mLvl),MedShk*np.ones_like(mLvl))
    plt.plot(mLvl,v)
    for j in range(J):
        v = MyType.solution[0].policyFunc[h][j].ValueFuncCopay.vFuncNvrs(mLvl,p*np.ones_like(mLvl),MedShk*np.ones_like(mLvl))
        plt.plot(mLvl,v)    
    plt.show()
    
    print('Consumption function by coinsurance rate:')
    mLvl = np.linspace(0,10,200)
    J = len(MyType.solution[0].vFuncByContract[h])
    cLvl,MedLvl = MyType.solution[0].policyFunc[h][0].PolicyFuncFullPrice(mLvl,p*np.ones_like(mLvl),MedShk*np.ones_like(mLvl))
    plt.plot(mLvl,cLvl)
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
    
    print('Consumption function by medical need shock for one contract')
    mLvl = np.linspace(0,10,200)
    j = 0
    for i in range(MyType.MedShkDstn[0][h][0].size):
        MedShk = MyType.MedShkDstn[0][h][1][i]*np.ones_like(mLvl)
        cLvl,MedLvl = MyType.solution[0].policyFunc[h][j](mLvl,p*np.ones_like(mLvl),MedShk)
        plt.plot(mLvl,cLvl)
    plt.show()
    
    print('Medical care function by medical need shock for one contract')
    mLvl = np.linspace(0,10,200)
    j = 0
    for i in range(MyType.MedShkDstn[0][h][0].size):
        MedShk = MyType.MedShkDstn[0][h][1][i]*np.ones_like(mLvl)
        cLvl,MedLvl = MyType.solution[0].policyFunc[h][j](mLvl,p*np.ones_like(mLvl),MedShk)
        plt.plot(mLvl,MedLvl)
    plt.show()
    
        