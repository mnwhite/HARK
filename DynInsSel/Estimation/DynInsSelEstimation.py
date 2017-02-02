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
from LoadDataMoments import data_moments, moment_weights, UseOOPbool
from HARKinterpolation import ConstantFunction
from HARKutilities import approxUniform, getPercentiles
from HARKcore import Market
from HARKparallel import multiThreadCommands, multiThreadCommandsFake


class DynInsSelType(InsSelConsumerType):
    '''
    An extension of InsSelConsumerType that adds and adjusts methods for estimation.
    '''    
    def makeIncBoolArray(self):
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
        IncQuintBoolArray = np.zeros((self.T_sim,self.AgentCount,5),dtype=bool)
        for t in range(self.T_sim):
            for q in range(5):
                if q == 0:
                    bot = 0.0
                else:
                    bot = self.IncomeQuintiles[t,q-1]
                if q == 4:
                    top = np.inf
                else:
                    top = self.IncomeQuintiles[t,q]
                IncQuintBoolArray[t,:,q] = np.logical_and(self.pLvlHist[t,:] >= bot, self.pLvlHist[t,:] < top)
        self.IncQuintBoolArray = IncQuintBoolArray
        #plt.plot(np.cumsum(np.sum(IncQuintBoolArray,axis=1),axis=1))
        #plt.show()
        
    def preSolve(self):
        self.updateSolutionTerminal()
        
    def initializeSim(self):
        InsSelConsumerType.initializeSim(self)
        if 'ContractNow' in self.track_vars:
            self.ContractNow_hist = np.zeros_like(self.ContractNow_hist).astype(int)
            
    def postSim(self):
        if self.UseOOPbool:
            self.MedHist = self.OOPnow_hist
        else:
            MedPrice_temp = np.tile(np.reshape(self.MedPrice[0:self.T_sim],(self.T_sim,1)),(1,self.AgentCount))
            self.MedHist = self.MedLvlNow_hist*MedPrice_temp
        self.WealthRatioHist = self.aLvlNow_hist/self.pLvlHist
        self.InsuredBoolArray = self.ContractNow_hist > 0
        
    def deleteSolution(self):
        del self.solution
    
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
        LogMedMeanByAge = np.zeros(60) + np.nan
        LogMedStdByAge = np.zeros(60) + np.nan
        InsuredRateByAge = np.zeros(40) + np.nan
        ZeroSubsidyRateByAge = np.zeros(40) + np.nan
        PremiumMeanByAge = np.zeros(40) + np.nan
        PremiumStdByAge = np.zeros(40) + np.nan

        # Calculate all simulated moments by age
        for t in range(60):
            WealthRatioList = []
            LogMedList = []
            PremiumList = []
            LiveCount = 0.0
            InsuredCount = 0.0
            ZeroCount = 0.0
            for ThisType in Agents:
                these = ThisType.LiveBoolArray[t,:]
                LogMedList.append(np.log(ThisType.MedHist[t,these]+0.0001))
                if t < 40:
                    WealthRatioList.append(ThisType.WealthRatioHist[t,these])
                    these = ThisType.InsuredBoolArray[t,:]
                    PremiumList.append(ThisType.PremNow_hist[t,these])
                    LiveCount += np.sum(ThisType.LiveBoolArray[t,:])
                    temp = np.sum(ThisType.InsuredBoolArray[t,:])
                    InsuredCount += temp
                    if ThisType.ZeroSubsidyBool:
                        ZeroCount += temp
            if t < 40:
                WealthRatioArray = np.hstack(WealthRatioList)
                WealthMedianByAge[t] = np.median(WealthRatioArray)
                PremiumArray = np.hstack(PremiumList)
                PremiumMeanByAge[t] = np.mean(PremiumArray)
                PremiumStdByAge[t] = np.std(PremiumArray)
                InsuredRateByAge[t] = InsuredCount/LiveCount
                if InsuredCount > 0.0:
                    ZeroSubsidyRateByAge[t] = ZeroCount/InsuredCount
                else:
                    ZeroSubsidyRateByAge[t] = 0.0 # This only happens with no insurance choice
            LogMedArray = np.hstack(LogMedList)
            LogMedMeanByAge[t] = np.mean(LogMedArray)
            LogMedStdByAge[t] = np.std(LogMedArray)
            
        # Initialize moments by age-health and define age band bounds
        LogMedMeanByAgeHealth = np.zeros((12,5)) + np.nan
        LogMedStdByAgeHealth = np.zeros((12,5)) + np.nan
        AgeBounds = [[0,5],[5,10],[10,15],[15,20],[20,25],[25,30],[30,35],[35,40],[40,45],[45,50],[50,55],[55,60]]

        # Calculate all simulated moments by age-health
        for a in range(12):
            bot = AgeBounds[a][0]
            top = AgeBounds[a][1]
            for h in range(5):
                LogMedList = []
                for ThisType in Agents:
                    these = ThisType.HealthBoolArray[bot:top,:,h]
                    LogMedList.append(np.log(ThisType.MedHist[bot:top,:][these]+0.0001))
                LogMedArray = np.hstack(LogMedList)
                LogMedMeanByAgeHealth[a,h] = np.mean(LogMedArray)
                LogMedStdByAgeHealth[a,h] = np.std(LogMedArray)
                
        # Initialize moments by age-income
        WealthMedianByAgeIncome = np.zeros((8,5)) + np.nan
        LogMedMeanByAgeIncome = np.zeros((8,5)) + np.nan
        LogMedStdByAgeIncome = np.zeros((8,5)) + np.nan
        InsuredRateByAgeIncome = np.zeros((8,5)) + np.nan
        PremiumMeanByAgeIncome = np.zeros((8,5)) + np.nan

        # Calculated all simulated moments by age-income
        for a in range(8):
            bot = AgeBounds[a][0]
            top = AgeBounds[a][1]
            for i in range(5):
                WealthRatioList = []
                LogMedList = []
                PremiumList = []
                LiveCount = 0.0
                InsuredCount = 0.0
                for ThisType in Agents:
                    these = ThisType.IncQuintBoolArray[bot:top,:,i]
                    LiveCount += np.sum(ThisType.LiveBoolArray[bot:top,:][these])
                    LogMedList.append(np.log(ThisType.MedHist[bot:top,:][these]+0.0001))
                    WealthRatioList.append(ThisType.WealthRatioHist[bot:top,:][these])
                    these = np.logical_and(ThisType.InsuredBoolArray[bot:top,:],these)
                    PremiumList.append(ThisType.PremNow_hist[bot:top,:][these])                    
                    temp = np.sum(ThisType.InsuredBoolArray[bot:top,:][these])
                    InsuredCount += temp
                WealthRatioArray = np.hstack(WealthRatioList)
                WealthMedianByAgeIncome[a,i] = np.median(WealthRatioArray)
                PremiumArray = np.hstack(PremiumList)
                PremiumMeanByAgeIncome[a,i] = np.mean(PremiumArray)
                InsuredRateByAgeIncome[a,i] = InsuredCount/LiveCount
                LogMedArray = np.hstack(LogMedList)
                LogMedMeanByAgeIncome[a,i] = np.mean(LogMedArray)
                LogMedStdByAgeIncome[a,i] = np.std(LogMedArray)
        
        # Store all of the simulated moments as attributes of self
        self.WealthMedianByAge = WealthMedianByAge
        self.LogMedMeanByAge = LogMedMeanByAge + 9.21
        self.LogMedStdByAge = LogMedStdByAge
        self.InsuredRateByAge = InsuredRateByAge
        self.ZeroSubsidyRateByAge = ZeroSubsidyRateByAge
        self.PremiumMeanByAge = PremiumMeanByAge
        self.PremiumStdByAge = PremiumStdByAge
        self.LogMedMeanByAgeHealth = LogMedMeanByAgeHealth + 9.21
        self.LogMedStdByAgeHealth = LogMedStdByAgeHealth
        self.WealthMedianByAgeIncome = WealthMedianByAgeIncome
        self.LogMedMeanByAgeIncome = LogMedMeanByAgeIncome + 9.21
        self.LogMedStdByAgeIncome = LogMedStdByAgeIncome
        self.InsuredRateByAgeIncome = InsuredRateByAgeIncome
        self.PremiumMeanByAgeIncome = PremiumMeanByAgeIncome
        
    def combineSimulatedMoments(self):
        '''
        Creates a single 1D array with all simulated moments, stored as an
        attribute of self.  Should only be run after calcSimulatedMoments().
        
        Parameters
        ----------
        None
        
        Returns
        -------
        None
        '''
        MomentList = [self.WealthMedianByAge,
                      self.LogMedMeanByAge,
                      self.LogMedStdByAge,
                      self.InsuredRateByAge,
                      self.ZeroSubsidyRateByAge,
                      self.PremiumMeanByAge,
                      self.PremiumStdByAge,
                      self.LogMedMeanByAgeHealth.flatten(),
                      self.LogMedStdByAgeHealth.flatten(),
                      self.WealthMedianByAgeIncome.flatten(),
                      self.LogMedMeanByAgeIncome.flatten(),
                      self.LogMedStdByAgeIncome.flatten(),
                      self.InsuredRateByAgeIncome.flatten(),
                      self.PremiumMeanByAgeIncome.flatten()]
        self.simulated_moments = np.hstack(MomentList)
        
    def aggregateMomentConditions(self):
        '''
        Calculates the overall "model fit" at the current parameters by adding
        up the weighted sum of squared moments.  The market should already have
        run the combineSimulatedMoments() method, and should also have the
        attributes data_moments and moment_weights defined (usually at init).
        
        Parameters
        ----------
        None
        
        Returns
        -------
        moment_sum : float
             Weighted sum of moment conditions.
        '''
        moment_conditions = np.log(self.simulated_moments/self.data_moments)
        moment_conditions[self.moment_weights == 0.0] = 0.0
        moment_sum = np.dot(moment_conditions**2,self.moment_weights)
        self.moment_sum = moment_sum
        return moment_sum
        
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
        IncomeQuintiles = np.zeros((self.Agents[0].T_sim,Cuts.size))
        
        # Get income quintile cut points for each age
        for t in range(self.Agents[0].T_sim):
            pLvlList = []
            for ThisType in self.Agents:
                pLvlList.append(ThisType.pLvlHist[t,:][ThisType.LiveBoolArray[t,:]])
            pLvlArray = np.hstack(pLvlList)
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
    MedShkMeanArray[4,:] = MedShkMeanEXfunc(AgeArray)
    MedShkMeanArray[3,:] = MedShkMeanVGfunc(AgeArray)
    MedShkMeanArray[2,:] = MedShkMeanGDfunc(AgeArray)
    MedShkMeanArray[1,:] = MedShkMeanFRfunc(AgeArray)
    MedShkMeanArray[0,:] = MedShkMeanPRfunc(AgeArray)
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
        WorkingContractList.append(MedInsuranceContract(ConstantFunction(0.0),0.05,0.05,Params.MedPrice))
    RetiredContractListA = [MedInsuranceContract(ConstantFunction(0.0),0.0,0.05,Params.MedPrice)]
    #RetiredContractListB = [MedInsuranceContract(ConstantFunction(0.0),0.2,0.05,Params.MedPrice)]
    TypeDict['ContractList'] = Params.working_T*[5*[WorkingContractList]] + (Params.retired_T)*[5*[RetiredContractListA]]
    
    # Make and return a DynInsSelType
    ThisType = DynInsSelType(**TypeDict)
    ThisType.track_vars = ['aLvlNow','cLvlNow','MedLvlNow','PremNow','ContractNow','OOPnow']
    ThisType.UseOOPbool = UseOOPbool
    if PremiumSubsidy == 0.0:
        ThisType.ZeroSubsidyBool = True
    else:
        ThisType.ZeroSubsidyBool = False
    return ThisType
        

def makeMarketFromParams(ParamArray,PremiumArray,InsChoiceBool):
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
            AgentList[-1].AgentCount = int(round(AgentList[-1].Weight*Params.AgentCountTotal))

    # Make a market to hold the agents
    InsuranceMarket = DynInsSelMarket()
    InsuranceMarket.Agents = AgentList
    InsuranceMarket.data_moments = data_moments
    InsuranceMarket.moment_weights = moment_weights
    return InsuranceMarket
    

def objectiveFunction(Parameters):
    InsChoice = False
    MyMarket = makeMarketFromParams(Parameters,np.array([1,2,3,4,5]),InsChoice)
    multiThreadCommandsFake(MyMarket.Agents,['update()','makeShockHistory()'])
    MyMarket.getIncomeQuintiles()
    multiThreadCommandsFake(MyMarket.Agents,['makeIncBoolArray()'])
    
    all_commands = ['update()','solve()','initializeSim()','simulate()','postSim()','deleteSolution()']
    multiThreadCommands(MyMarket.Agents,all_commands)
    
    MyMarket.calcSimulatedMoments()
    MyMarket.combineSimulatedMoments()
    moment_sum = MyMarket.aggregateMomentConditions()    
    return MyMarket

    
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from time import clock
    mystr = lambda number : "{:.4f}".format(number)
    
#    t_start = clock()
#    InsChoice = False
#    MyMarket = makeMarketFromParams(Params.test_param_vec,np.array([1,2,3,4,5]),InsChoice)
#    multiThreadCommandsFake(MyMarket.Agents,['update()','makeShockHistory()'])
#    MyMarket.getIncomeQuintiles()
#    multiThreadCommandsFake(MyMarket.Agents,['makeIncBoolArray()'])
#    t_end = clock()
#    print('Making the agents took ' + mystr(t_end-t_start) + ' seconds.')
    
#    MyMarket.Agents[2].solve()

#    t_start = clock()
#    solve_commands = ['solve()']
#    multiThreadCommands(MyMarket.Agents,solve_commands)
#    t_end = clock()
#    print('Solving the agents took ' + mystr(t_end-t_start) + ' seconds.')
#    
#    t_start = clock()
#    sim_commands = ['initializeSim()','simulate()','postSim()']
#    multiThreadCommands(MyMarket.Agents,sim_commands)
#    t_end = clock()
#    print('Simulating agents took ' + mystr(t_end-t_start) + ' seconds.')
    
#    t_start = clock()
#    all_commands = ['update()','solve()','initializeSim()','simulate()','postSim()']
#    multiThreadCommands(MyMarket.Agents,all_commands)
#    t_end = clock()
#    print('Solving and simulating agents took ' + mystr(t_end-t_start) + ' seconds.')
#    
#    MyMarket.calcSimulatedMoments()

    t_start = clock()
    MyMarket = objectiveFunction(Params.test_param_vec)
    t_end = clock()
    print('Objective function evaluation took ' + mystr(t_end-t_start) + ' seconds.')
    
    MyType = MyMarket.Agents[0]    
    t = 0
    p = 2.0    
    h = 4        
    MedShk = 1.0e-2
    z = 0
    
#    MyType.plotvFunc(t,p)
#    MyType.plotvPfunc(t,p)
#    MyType.plotvFuncByContract(t,h,p)
#    MyType.plotcFuncByContract(t,h,p,MedShk)
#    MyType.plotcFuncByMedShk(t,h,z,p)
#    MyType.plotMedFuncByMedShk(t,h,z,p)
    
    plt.plot(MyMarket.WealthMedianByAge)
    plt.plot(MyMarket.data_moments[0:40],'.k')
    plt.show()
    
    plt.plot(MyMarket.LogMedMeanByAge)
    plt.plot(MyMarket.data_moments[40:100],'.k')
    plt.show()
    
    plt.plot(MyMarket.LogMedStdByAge)
    plt.plot(MyMarket.data_moments[100:160],'.k')
    plt.show()
    
    plt.plot(MyMarket.LogMedMeanByAgeHealth)
    temp = np.reshape(MyMarket.data_moments[320:380],(12,5))
    plt.plot(temp,'.')
    plt.show()
    
    plt.plot(MyMarket.LogMedStdByAgeHealth)
    plt.show()
    
        