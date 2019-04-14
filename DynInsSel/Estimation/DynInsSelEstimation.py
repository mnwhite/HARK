'''
This is a first attempt at the DynInsSel estimation.  It has functions to construct agent type(s)
from input parameters; it will eventually have the ability to compare simulated data to real data.
'''
import sys 
sys.path.insert(0,'../../')

import numpy as np
import DynInsSelParameters as Params
from time import clock
from copy import copy, deepcopy
from InsuranceSelectionModel import MedInsuranceContract, InsSelConsumerType, InsSelStaticConsumerType
from LoadDataMoments import data_moments, moment_weights
from ActuarialRules import flatActuarialRule, ageHealthRatedActuarialRule, InsuranceMarket
from HARKinterpolation import ConstantFunction
from HARKutilities import getPercentiles
from HARKparallel import multiThreadCommands, multiThreadCommandsFake

if Params.StaticBool:
    BaseType = InsSelStaticConsumerType
else:
    BaseType = InsSelConsumerType

class DynInsSelType(BaseType):
    '''
    An extension of InsSelConsumerType that adds and adjusts methods for estimation.
    '''    
    def makeIncBoolArray(self):
        '''
        Makes the attribute IncQuintBoolArray, specifying which income quintile
        each agent is in for each period of life.  Should only be run after the
        economy runs the method getIncomeQuintiles().
        
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
        
    def preSolve(self):
        '''
        Moves the premium functions from the attribute PremiumFuncs into the
        Premium attribute in ContractList, so that they are used correctly
        during solution of the dynamic model.
        '''
        self.installPremiumFuncs()
        self.updateSolutionTerminal()
        
    def postSolve(self): # This needs to be active in the dynamic model
        '''
        Initializes and executes the simulation, and does some post-simulation
        processing of the results.  Also deletes most of the dynamic solution
        so that a smaller pickled object is passed back to the master process,
        using less hard drive time.  This method should be different if the
        quasi-static model is ever used, but I forget how at this point.
        '''
        t0 = clock()
        if self.do_sim:
            self.initializeSim()
            self.simulate()
            self.postSim()
        if self.del_soln:
            self.deleteSolution()
        t1 = clock()
        if self.verbosity > 0:
            print('Simulating this agent type took ' + str(t1 - t0) + ' seconds.')
        
    def initializeSim(self):
        '''
        Initializes the simulation using the base class method, making sure that
        ContractNow is tracked as an integer, not a float.
        '''
        InsSelConsumerType.initializeSim(self)
        if 'ContractNow' in self.track_vars:
            self.ContractNow_hist = np.zeros_like(self.ContractNow_hist).astype(int)
            
    def postSim(self):
        '''
        Makes small fixes to OOP medical spending (can't be below $1 but above $0),
        and calculates total medical spending, wealth-to-income ratio, and insured status.
        '''
        self.OOPmedHist = self.OOPnow_hist
        self.OOPmedHist[np.logical_and(self.OOPmedHist < 0.0001,self.OOPmedHist > 0.0,)] = 0.0001
        MedPrice_temp = np.tile(np.reshape(self.MedPrice[0:self.T_sim],(self.T_sim,1)),(1,self.AgentCount))
        self.TotalMedHist = self.MedLvlNow_hist*MedPrice_temp
        self.TotalMedHist[np.logical_and(self.TotalMedHist < 0.0001,self.TotalMedHist > 0.0,)] = 0.0001
        self.WealthRatioHist = self.aLvlNow_hist/self.pLvlHist
        self.InsuredBoolArray = self.ContractNow_hist > 0
        
    def deleteSolution(self):
        '''
        Deletes most of the dynamic solution in order to reduce the size of the
        objected that is pickled back to the master process by joblib.  Keeps
        the value function (overall and by contract) and the actuarial value
        functions, as these are needed during static premium equilibrium search
        and counterfactual analysis.
        '''
        vFuncByContract = []
        AVfunc = []
        vFunc = []
        for t in range(self.T_cycle):
            vFuncByContract.append(self.solution[t].vFuncByContract)
            AVfunc.append(self.solution[t].AVfunc)
            vFunc.append(self.solution[t].vFunc)
        self.vFuncByContract = vFuncByContract
        self.AVfunc = AVfunc
        self.vFunc = vFunc
        self.addToTimeVary('AVfunc','vFuncByContract','vFunc')
        del self.solution
        self.delFromTimeVary('solution')
        
    def reset(self): # Apparently nothing at all
        return None
    
    def makevHistArray(self):
        '''
        Create an array of value levels based on the simulated history of this
        type, stored as the attribute vNow_hist.  This is called only during
        counterfactual analysis (of the "before" world).
        '''
        N = self.AgentCount
        T = 60
        vNow_hist = np.zeros((T,N)) + np.nan
        for t in range(T):
            StateCount = len(self.vFunc[t])
            for h in range(StateCount):
                these = self.HealthBoolArray[t,:,h]
                m_temp = self.mLvlNow_hist[t,these]
                p_temp = self.pLvlHist[t,these]
                vNow_hist[t,these] = self.vFunc[t][h](m_temp,p_temp)
        self.vNow_hist = vNow_hist
        
        
    def findCompensatingpLvl(self):
        '''
        For each simulated agent in the history, find the pLvl that makes them
        indifferent between the current state of the world (the "after" scenario)
        and a target value level (from the "before" scenario).  This is called
        only during counterfactual analysis (of the "after" world).
        '''
        N = self.AgentCount
        T = 60
        pCompHist = np.zeros((T,N)) + np.nan
        out_of_bounds = np.zeros((T,N),dtype=bool) # Mark agents whose pComp is too high (requires extrapolation)
        for t in range(T):
            StateCount = len(self.vFunc[t])
            for h in range(StateCount):
                # Extract data for this age-health
                these = self.HealthBoolArray[t,:,h]
                m = self.mLvlNow_hist[t,these]
                vTarg = self.vTarg_hist[t,these]
                vFunc = self.vFunc[t][h]
                pMin = vFunc.func.func.y_list[0] # Lowest pLvl, should always be 0.0
                pMax = vFunc.func.func.y_list[-1] # Highest pLvl in grid
                 
                # Evaluate the value function at the minimum and maximum pLvl
                pBot = pMin*np.ones(np.sum(these))
                pTop = pMax*np.ones(np.sum(these))
                vBot = vFunc(m,pBot)
                vTop = vFunc(m,pTop)
                out_of_bounds[t,these] = np.logical_or(vTarg < vBot, vTarg > vTop)
                
                # Do a bracketing search for compensating pLvl
                j_max = 20 # Number of loops to run
                for j in range(j_max):
                    pMid = 0.5*(pBot + pTop)
                    vMid = vFunc(m,pMid)
                    targ_below_mid = vTarg <= vMid
                    targ_above_mid = np.logical_not(targ_below_mid)
                    pTop[targ_below_mid] = pMid[targ_below_mid]
                    pBot[targ_above_mid] = pMid[targ_above_mid]
                    
                # Record the result of the bracketing search in pComp_hist
                pCompHist[t,these] = pMid
                
        # Store the results as attributes of self
        self.pCompHist = pCompHist
        self.pComp_invalid = out_of_bounds
        
        
class DynInsSelMarket(InsuranceMarket):
    '''
    A class for representing the "insurance economy" with many agent types, with
    methods specific to the DynInsSel project: calculating simulated moments, etc.
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
        Agents = self.agents
        
        # Initialize moments by age
        WealthMedianByAge = np.zeros(40) + np.nan
        LogTotalMedMeanByAge = np.zeros(60) + np.nan
        LogTotalMedStdByAge = np.zeros(60) + np.nan
        InsuredRateByAge_ESI = np.zeros(40) + np.nan
        InsuredRateByAge_IMI = np.zeros(40) + np.nan
        ZeroSubsidyRateByAge = np.zeros(40) + np.nan
        PremiumMeanByAge = np.zeros(40) + np.nan
        PremiumStdByAge = np.zeros(40) + np.nan
        LogOOPmedMeanByAge = np.zeros(60) + np.nan
        LogOOPmedStdByAge = np.zeros(60) + np.nan
        OOPshareByAge = np.zeros(60) + np.nan

        # Calculate all simulated moments by age
        for t in range(60):
            WealthRatioList = []
            LogTotalMedList = []
            LogOOPmedList = []
            PremiumList = []
            TotalMedSum = 0.0
            OOPmedSum = 0.0
            LiveCount = 0.0
            ESIcount = 0.0
            OfferedCount = 0.0
            IMIcount = 0.0
            NotOfferedCount = 0.0
            ZeroCount = 0.0
            for ThisType in Agents:
                these = ThisType.LiveBoolArray[t,:]
                those = np.logical_and(these,ThisType.TotalMedHist[t,:]>0.0)
                LogTotalMedList.append(np.log(ThisType.TotalMedHist[t,those]))
                LogOOPmedList.append(np.log(ThisType.OOPmedHist[t,those]))
                if t < 40:
                    WealthRatioList.append(ThisType.WealthRatioHist[t,these])
                    Insured = ThisType.InsuredBoolArray[t,:]
                    LiveCount += np.sum(ThisType.LiveBoolArray[t,:])
                    Offered = ThisType.MrkvHist[t,:] >= 5
                    NotOffered = np.logical_and(ThisType.MrkvHist[t,:] < 5, ThisType.MrkvHist[t,:] >= 0) 
                    HaveESI = np.logical_and(Insured, Offered)
                    ESIcount += np.sum(HaveESI)
                    OfferedCount += np.sum(Offered)
                    IMIcount += np.sum(np.logical_and(Insured,NotOffered))
                    NotOfferedCount += np.sum(NotOffered)
                    ZeroCount += np.sum(np.logical_and(HaveESI, ThisType.MrkvHist[t,:] <= 9))
                    PremiumList.append(ThisType.PremNow_hist[t,HaveESI])
                    TotalMedSum += np.sum(ThisType.TotalMedHist[t,HaveESI])
                    OOPmedSum += np.sum(ThisType.OOPmedHist[t,HaveESI])
                else:
                    TotalMedSum += np.sum(ThisType.TotalMedHist[t,these])
                    OOPmedSum += np.sum(ThisType.OOPmedHist[t,these])
            if t < 40:
                WealthRatioArray = np.hstack(WealthRatioList)
                WealthMedianByAge[t] = np.median(WealthRatioArray)
                PremiumArray = np.hstack(PremiumList)
                PremiumMeanByAge[t] = np.mean(PremiumArray)
                PremiumStdByAge[t] = np.std(PremiumArray)
                InsuredRateByAge_ESI[t] = ESIcount/OfferedCount
                InsuredRateByAge_IMI[t] = IMIcount/NotOfferedCount
                if ESIcount > 0.0:
                    ZeroSubsidyRateByAge[t] = ZeroCount/ESIcount
                else:
                    ZeroSubsidyRateByAge[t] = 0.0 # This only happens with no insurance choice
            LogTotalMedArray = np.hstack(LogTotalMedList)
            LogTotalMedMeanByAge[t] = np.mean(LogTotalMedArray)
            LogTotalMedStdByAge[t] = np.std(LogTotalMedArray)
            LogOOPmedArray = np.hstack(LogOOPmedList)
            LogOOPmedMeanByAge[t] = np.mean(LogOOPmedArray)
            LogOOPmedStdByAge[t] = np.std(LogOOPmedArray)
            OOPshareByAge[t] = OOPmedSum/TotalMedSum
            
        # Initialize moments by age-health and define age band bounds
        LogTotalMedMeanByAgeHealth = np.zeros((12,5)) + np.nan
        LogTotalMedStdByAgeHealth = np.zeros((12,5)) + np.nan
        AgeBounds = [[0,5],[5,10],[10,15],[15,20],[20,25],[25,30],[30,35],[35,40],[40,45],[45,50],[50,55],[55,60]]

        # Calculate all simulated moments by age-health
        for a in range(12):
            bot = AgeBounds[a][0]
            top = AgeBounds[a][1]
            for h in range(5):
                LogTotalMedList = []
                for ThisType in Agents:
                    these = ThisType.HealthBoolArray[bot:top,:,h]
                    those = np.logical_and(these,ThisType.TotalMedHist[bot:top,:]>0.0)
                    LogTotalMedList.append(np.log(ThisType.TotalMedHist[bot:top,:][those]))
                LogTotalMedArray = np.hstack(LogTotalMedList)
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
            bot = AgeBounds[a][0]
            top = AgeBounds[a][1]
            for i in range(5):
                WealthRatioList = []
                LogTotalMedList = []
                PremiumList = []
                LiveCount = 0.0
                ESIcount = 0.0
                OfferedCount = 0.0
                for ThisType in Agents:
                    these = ThisType.IncQuintBoolArray[bot:top,:,i]
                    those = np.logical_and(these,ThisType.TotalMedHist[bot:top,:] > 0.0)
                    LiveCount += np.sum(ThisType.LiveBoolArray[bot:top,:][these])
                    Offered = ThisType.MrkvHist[bot:top,:][these] >= 5
                    OfferedCount += np.sum(Offered)
                    LogTotalMedList.append(np.log(ThisType.TotalMedHist[bot:top,:][those]))
                    WealthRatioList.append(ThisType.WealthRatioHist[bot:top,:][these])
                    HaveESI = np.logical_and(ThisType.InsuredBoolArray[bot:top,:][these],Offered)
                    ESIcount += np.sum(HaveESI)
                    PremiumList.append(ThisType.PremNow_hist[bot:top,:][these][HaveESI])                    
                WealthRatioArray = np.hstack(WealthRatioList)
                WealthMedianByAgeIncome[a,i] = np.median(WealthRatioArray)
                PremiumArray = np.hstack(PremiumList)
                PremiumMeanByAgeIncome[a,i] = np.mean(PremiumArray)
                InsuredRateByAgeIncome[a,i] = ESIcount/OfferedCount
                LogTotalMedArray = np.hstack(LogTotalMedList)
                LogTotalMedMeanByAgeIncome[a,i] = np.mean(LogTotalMedArray)
                LogTotalMedStdByAgeIncome[a,i] = np.std(LogTotalMedArray)
        
        # Store all of the simulated moments as attributes of self
        self.WealthMedianByAge = WealthMedianByAge
        self.LogTotalMedMeanByAge = LogTotalMedMeanByAge + 9.21034
        self.LogTotalMedStdByAge = LogTotalMedStdByAge
        self.InsuredRateByAge_ESI = InsuredRateByAge_ESI
        self.InsuredRateByAge_IMI = InsuredRateByAge_IMI
        self.ZeroSubsidyRateByAge = ZeroSubsidyRateByAge
        self.PremiumMeanByAge = PremiumMeanByAge*10000.
        self.PremiumStdByAge = PremiumStdByAge*10000.
        self.LogTotalMedMeanByAgeHealth = LogTotalMedMeanByAgeHealth + 9.21034
        self.LogTotalMedStdByAgeHealth = LogTotalMedStdByAgeHealth
        self.WealthMedianByAgeIncome = WealthMedianByAgeIncome
        self.LogTotalMedMeanByAgeIncome = LogTotalMedMeanByAgeIncome + 9.21034
        self.LogTotalMedStdByAgeIncome = LogTotalMedStdByAgeIncome
        self.InsuredRateByAgeIncome = InsuredRateByAgeIncome
        self.PremiumMeanByAgeIncome = PremiumMeanByAgeIncome*10000.
        self.LogOOPmedMeanByAge = LogOOPmedMeanByAge + 9.21034
        self.LogOOPmedStdByAge = LogOOPmedStdByAge
        self.OOPshareByAge = OOPshareByAge
        
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
                      self.LogTotalMedMeanByAge,
                      self.LogTotalMedStdByAge,
                      self.InsuredRateByAge_ESI,
                      self.ZeroSubsidyRateByAge,
                      self.PremiumMeanByAge,
                      self.PremiumStdByAge,
                      self.LogTotalMedMeanByAgeHealth.flatten(),
                      self.LogTotalMedStdByAgeHealth.flatten(),
                      self.WealthMedianByAgeIncome.flatten(),
                      self.LogTotalMedMeanByAgeIncome.flatten(),
                      self.LogTotalMedStdByAgeIncome.flatten(),
                      self.InsuredRateByAgeIncome.flatten(),
                      self.PremiumMeanByAgeIncome.flatten(),
                      self.LogOOPmedMeanByAge,
                      self.LogOOPmedStdByAge,
                      self.OOPshareByAge,
                      self.InsuredRateByAge_IMI]
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
        IncomeQuintiles = np.zeros((self.agents[0].T_sim,Cuts.size))
        
        # Get income quintile cut points for each age
        for t in range(self.agents[0].T_sim):
            pLvlList = []
            for ThisType in self.agents:
                pLvlList.append(ThisType.pLvlHist[t,:][ThisType.LiveBoolArray[t,:]])
            pLvlArray = np.hstack(pLvlList)
            IncomeQuintiles[t,:] = getPercentiles(pLvlArray,percentiles = Cuts)
        
        # Store the income quintile cut points in each AgentType
        for ThisType in self.agents:
            ThisType.IncomeQuintiles = IncomeQuintiles
                    
                               
def makeDynInsSelType(CRRAcon,MedCurve,DiscFac,BequestShift,BequestScale,Cfloor,ChoiceShkMag,
                      MedShkMeanAgeParams,MedShkMeanVGparams,MedShkMeanGDparams,MedShkMeanFRparams,MedShkMeanPRparams,
                      MedShkStdAgeParams,MedShkStdVGparams,MedShkStdGDparams,MedShkStdFRparams,MedShkStdPRparams,
                      PremiumSubsidy,EducType,InsChoiceType,ContractCount):
    '''
    Makes an InsSelConsumerType using (human-organized) structural parameters for the estimation.
    
    Parameters
    ----------
    CRRAcon : float
        Coefficient of relative risk aversion for consumption.
    MedCurve : float
        Coefficient of relative risk aversion for medical care.
    DiscFac : float
        Intertemporal discount factor.
    BequestShift : float
        Shifter in bequest motive function.
    BequestScale : float
        Scale of bequest motive function.
    Cfloor : float
        Minimum level of consumption that government will allow individual to suffer.
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
    PremiumSubsidy : float
        Employer contribution to any (non-null) insurance contract.
    EducType : int
        Discrete education type.  0 --> dropout, 1 --> high school, 2 --> college graduate.
    InsChoiceType : int
        Indicator for whether insurance choice is in the model.  When 2, working-age agents can
        choose among several insurance contracts, while retirees choose between basic Medicare,
        Medicare A+B, or Medicare A+B + Medigap.  When 0, agents get exogenous contracts.
        When 1, agents have a binary contract choice when working.
    ContractCount : int
        Total number of contracts available to agents, including null contract.
    
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
        
    # Make a timepath of discount factors
    DiscFac_time_vary = np.linspace(DiscFac-0.00,DiscFac,25).tolist() + 70*[DiscFac]  
        
    TypeDict['CRRA'] = CRRAcon
    TypeDict['MedCurve'] = MedCurve
    TypeDict['BequestShift'] = BequestShift
    TypeDict['BequestScale'] = BequestScale
    TypeDict['DiscFac'] = DiscFac_time_vary
    TypeDict['ChoiceShkMag'] = Params.AgeCount*[ChoiceShkMag]
    TypeDict['Cfloor'] = Cfloor
                          
    # Make arrays of medical shock means and standard deviations
    Taper = 1
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
    if Taper > 0:
        MedShkMeanEX85 = MedShkMeanEXfunc(60.)
        MedShkMeanVG85 = MedShkMeanVGfunc(60.)
        MedShkMeanGD85 = MedShkMeanGDfunc(60.)
        MedShkMeanFR85 = MedShkMeanFRfunc(60.)
        MedShkMeanPR85 = MedShkMeanPRfunc(60.)
        MedShkSlopeEX85 = MedShkMeanAgeParams[1] + 2*MedShkMeanAgeParams[2]*60 + 3*MedShkMeanAgeParams[3]*60**2 + 4*MedShkMeanAgeParams[4]*60**3
        MedShkSlopeVG85 = MedShkSlopeEX85 + MedShkMeanVGparams[1]
        MedShkSlopeGD85 = MedShkSlopeVG85 + MedShkMeanGDparams[1]
        MedShkSlopeFR85 = MedShkSlopeGD85 + MedShkMeanFRparams[1]
        MedShkSlopePR85 = MedShkSlopeFR85 + MedShkMeanPRparams[1]
        EX_A = -MedShkSlopeEX85/70.
        EX_B = 19./7.*MedShkSlopeEX85
        EX_C = MedShkMeanEX85 - 3600*EX_A - 60*EX_B        
        VG_A = -MedShkSlopeVG85/70.
        VG_B = 19./7.*MedShkSlopeVG85
        VG_C = MedShkMeanVG85 - 3600*VG_A - 60*VG_B        
        GD_A = -MedShkSlopeGD85/70.
        GD_B = 19./7.*MedShkSlopeGD85
        GD_C = MedShkMeanGD85 - 3600*GD_A - 60*GD_B        
        FR_A = -MedShkSlopeFR85/70.
        FR_B = 19./7.*MedShkSlopeFR85
        FR_C = MedShkMeanFR85 - 3600*FR_A - 60*FR_B        
        PR_A = -MedShkSlopePR85/70.
        PR_B = 19./7.*MedShkSlopePR85
        PR_C = MedShkMeanPR85 - 3600*PR_A - 60*PR_B
        MedShkMeanArray[4,60:] = EX_A*AgeArray[60:]**2 + EX_B*AgeArray[60:] + EX_C
        MedShkMeanArray[3,60:] = VG_A*AgeArray[60:]**2 + VG_B*AgeArray[60:] + VG_C
        MedShkMeanArray[2,60:] = GD_A*AgeArray[60:]**2 + GD_B*AgeArray[60:] + GD_C
        MedShkMeanArray[1,60:] = FR_A*AgeArray[60:]**2 + FR_B*AgeArray[60:] + FR_C
        MedShkMeanArray[0,60:] = PR_A*AgeArray[60:]**2 + PR_B*AgeArray[60:] + PR_C
                
    if Taper > 1:
        MedShkMeanArray[:,60:] = np.tile(np.reshape(MedShkMeanArray[:,60],(5,1)),(1,35)) # Hold distribution constant after age 85
        MedShkStdArray[:,60:] = np.tile(np.reshape(MedShkStdArray[:,60],(5,1)),(1,35))
    TypeDict['MedShkAvg'] = MedShkMeanArray.transpose().tolist()
    TypeDict['MedShkStd'] = MedShkStdArray.transpose().tolist()
    
    # Make insurance contracts when working and retired, then combine into a lifecycle list
    WorkingContractList = []
    if InsChoiceType >= 2:
        WorkingContractList.append(MedInsuranceContract(ConstantFunction(0.0),0.0,1.0,Params.MedPrice))
        for j in range(ContractCount-1):
            #Premium = max([PremiumArray[j] - PremiumSubsidy,0.0])
            Premium = 0.0 # Will be changed by installPremiumFuncs
            Copay = 0.08
            Deductible = Params.DeductibleList[j]
            WorkingContractList.append(MedInsuranceContract(ConstantFunction(Premium),Deductible,Copay,Params.MedPrice))
    elif InsChoiceType == 1:
        WorkingContractList.append(MedInsuranceContract(ConstantFunction(0.0),0.0,1.0,Params.MedPrice))
        Premium = 0. # Will be changed by installPremiumFuncs
        Copay = 0.08
        Deductible = 0.04
        WorkingContractList.append(MedInsuranceContract(ConstantFunction(Premium),Deductible,Copay,Params.MedPrice))
    else:
        WorkingContractList.append(MedInsuranceContract(ConstantFunction(0.0),0.00,0.1,Params.MedPrice))
    
    IndMarketContractList = [MedInsuranceContract(ConstantFunction(0.0),0.0,1.0,Params.MedPrice),
                             MedInsuranceContract(ConstantFunction(Premium),0.2,Copay,Params.MedPrice)]
    RetiredContractList = [MedInsuranceContract(ConstantFunction(0.0),0.0,0.12,Params.MedPrice)]
    
    ContractList = []
    for t in range(Params.working_T):
        ContractList_t = []
        for h in range(5): # distribute individual market contracts
            ContractList_t.append(deepcopy(IndMarketContractList))
        for h in range(10): # distribute ESI contracts
            ContractList_t.append(deepcopy(WorkingContractList))
        ContractList.append(ContractList_t)
    for t in range(Params.retired_T):
        ContractList_t = []
        for h in range(5):
            ContractList_t.append(deepcopy(RetiredContractList))
        ContractList.append(ContractList_t)
    
    TypeDict['ContractList'] = ContractList
    
    # Make and return a DynInsSelType
    ThisType = DynInsSelType(**TypeDict)
    ThisType.track_vars = ['aLvlNow','mLvlNow','cLvlNow','MedLvlNow','PremNow','ContractNow','OOPnow','WelfareNow']
    ThisType.PremiumSubsidy = PremiumSubsidy
    if PremiumSubsidy == 0.0:
        ThisType.ZeroSubsidyBool = True
    else:
        ThisType.ZeroSubsidyBool = False
    return ThisType
        

def makeMarketFromParams(ParamArray,ActuarialRule,IMIpremiumArray,ESIpremiumArray,InsChoiceType):
    '''
    Makes a list of DynInsSelTypes, to be used for estimation.
    
    Parameters
    ----------
    ParamArray : np.array
        Array of size 35, representing all of the structural parameters.
    ActuarialRule : function
        Function representing how insurance market outcomes are translated into
        premiums.  Will be installed as the millRule attribute of the market.
    IMIpremiumArray : np.array
        Array of premiums for IMI contracts for workers. Irrelevant if InsChoiceType = 0.
        Should be of size (40,ContractCount).
    ESIpremiumArray : np.array
        Array of premiums for ESI contracts for workers. Irrelevant if InsChoiceType = 0.
        Should be of size (40,ContractCount).
    InsChoiceType : int
        Indicator for the extent of consumer choice over contracts.  0 --> no choice,
        1 --> one non-null contract, 2 --> five non-null contracts.
    SubsidyTypeCount : int
        Number of different non-zero subsidy levels for consumers in this market.
    CRRAtypeCount : int
        Number of different CRRA values in the population.
    ZeroSubsidyBool : bool
        Indicator for whether there is a zero subsidy type.
        
    Returns
    -------
    ThisMarket : DynInsSelMarket
        Market to be used in estimation or counterfactual, with agents filled in.
    '''
    # Unpack the parameters
    DiscFac = ParamArray[0]
    CRRAcon = ParamArray[1]
    MedCurve = ParamArray[2]
    ChoiceShkMag = np.exp(ParamArray[3])
    Cfloor = ParamArray[4]
    EmpContr = np.exp(ParamArray[5])
    BequestShift = ParamArray[7]
    BequestScale = ParamArray[8]
    MedShkMeanAgeParams = ParamArray[9:14]
    MedShkMeanVGparams = ParamArray[14:16]
    MedShkMeanGDparams = ParamArray[16:18]
    MedShkMeanFRparams = ParamArray[18:20]
    MedShkMeanPRparams = ParamArray[20:22]
    MedShkStdAgeParams = ParamArray[22:27]
    MedShkStdVGparams = ParamArray[27:29]
    MedShkStdGDparams = ParamArray[29:31]
    MedShkStdFRparams = ParamArray[31:33]
    MedShkStdPRparams = ParamArray[33:35]
    
    # Make the list of types
    AgentList = []
    ContractCount = ESIpremiumArray.shape[1]
    i = 0
    for k in range(3):
        ThisAgent = makeDynInsSelType(CRRAcon,MedCurve,DiscFac,BequestShift,BequestScale,Cfloor,ChoiceShkMag,MedShkMeanAgeParams,
                  MedShkMeanVGparams,MedShkMeanGDparams,MedShkMeanFRparams,MedShkMeanPRparams,
                  MedShkStdAgeParams,MedShkStdVGparams,MedShkStdGDparams,MedShkStdFRparams,
                  MedShkStdPRparams,EmpContr,k,InsChoiceType,ContractCount)
        ThisAgent.Weight = Params.EducWeight[k]
        ThisAgent.AgentCount = int(round(ThisAgent.Weight*Params.AgentCountTotal))
        AgentList.append(ThisAgent)
        i += 1
    
    for i in range(len(AgentList)):
        AgentList[i].seed = i # Assign different seeds to each type
            
    # Construct an initial nested list for premiums
    PremiumFuncs_init = []
    HealthCount = 5
    for t in range(40):
        PremiumFuncs_t = []
        for h in range(HealthCount):
            PremiumFuncs_this_health = []
            for z in range(ContractCount):
                if z > 0:
                    if h == 0:
                        Prem = 10000.
                    else:
                        Prem = IMIpremiumArray[t]
                else:
                    Prem = 0.
                PremiumFuncs_this_health.append(ConstantFunction(Prem))
            PremiumFuncs_t.append(PremiumFuncs_this_health)
        for h in range(2*HealthCount):
            PremiumFuncs_this_health = []
            for z in range(ContractCount):
                Prem = ESIpremiumArray[t,z]
                PremiumFuncs_this_health.append(ConstantFunction(Prem))
            PremiumFuncs_t.append(PremiumFuncs_this_health)
        PremiumFuncs_init.append(PremiumFuncs_t)

    # Make a market to hold the agents
    ThisMarket = DynInsSelMarket(ActuarialRule)
    ThisMarket.agents = AgentList
    ThisMarket.data_moments = data_moments
    ThisMarket.moment_weights = moment_weights
    ThisMarket.PremiumFuncs_init = PremiumFuncs_init
    ThisMarket.LoadFacESI = 1.2 # Make this an input later
    ThisMarket.LoadFacIMI = 1.8
    
    # Have each agent type in the market inherit the premium functions
    for this_agent in ThisMarket.agents:
        setattr(this_agent,'PremiumFuncs',PremiumFuncs_init)
    
    #print('I made an insurance market with ' + str(len(InsuranceMarket.agents)) + ' agent types!')
    return ThisMarket


def objectiveFunction(Parameters):
    '''
    The objective function for the estimation.  Makes and solves a market, then
    returns the weighted sum of moment differences between simulation and data.
    '''
    EvalType  = 2  # Number of times to do a static search for eqbm premiums
    InsChoice = 1  # Extent of insurance choice
    TestPremiums = True # Whether to start with the test premium level
    
    if TestPremiums:
        ESIpremiums = np.array([0.3500, 0.0, 0.0, 0.0, 0.0])
    else:
        ESIpremiums = Params.PremiumsLast
    IMIpremiums_init = Params.IMIpremiums
    
    ContractCounts = [0,1,5] # plus one
    ESIpremiums_init_short = np.concatenate((np.array([0.]),ESIpremiums[0:ContractCounts[InsChoice]]))
    ESIpremiums_init = np.tile(np.reshape(ESIpremiums_init_short,(1,ESIpremiums_init_short.size)),(40,1))
    
    MyMarket = makeMarketFromParams(Parameters,ageHealthRatedActuarialRule,IMIpremiums_init,ESIpremiums_init,InsChoice)
    MyMarket.HealthGroups = [[0],[1,2,3,4]]
    MyMarket.ExcludedGroups = [True,False]
    MyMarket.ESIpremiums = ESIpremiums_init_short
    multiThreadCommandsFake(MyMarket.agents,['update()','makeShockHistory()'])
    MyMarket.getIncomeQuintiles()
    multiThreadCommandsFake(MyMarket.agents,['makeIncBoolArray()'])
    
    if EvalType == 0:
        multiThreadCommands(MyMarket.agents,['solve()'])
    else:
        MyMarket.max_loops = EvalType
        MyMarket.solve()
        Params.PremiumsLast = MyMarket.ESIpremiums

    MyMarket.calcSimulatedMoments()
    MyMarket.combineSimulatedMoments()
    moment_sum = MyMarket.aggregateMomentConditions()    
    return MyMarket


###############################################################################
    
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    mystr = lambda number : "{:.4f}".format(number)
    
    test_obj_func = True
    test_one_type = False
    test_static_model = False
    
    if test_obj_func:
    # This block is for actually testing the objective function
        t_start = clock()
        MyMarket = objectiveFunction(Params.test_param_vec)
        t_end = clock()
        print('Objective function evaluation took ' + mystr(t_end-t_start) + ' seconds.')
        
        # This block of code is for displaying moment fits after running objectiveFunc  
        Age = np.arange(25,85)
        Age5year = 27.5 + 5*np.arange(12)
        
        if not Params.StaticBool:
            plt.plot(Age[0:40],MyMarket.WealthMedianByAge)
            plt.plot(Age[0:40],MyMarket.data_moments[0:40],'.k')
            plt.xlabel('Age')
            plt.ylabel('Median wealth/income ratio')
            #plt.savefig('../Figures/WealthFitByAge.pdf')
            plt.show()
        
        plt.plot(Age,MyMarket.LogTotalMedMeanByAge)
        plt.plot(Age,MyMarket.data_moments[40:100],'.k')
        plt.xlabel('Age')
        plt.ylabel('Mean log total (nonzero) medical expenses')
        plt.xlim((25,85))
        #plt.savefig('../Figures/MeanTotalMedFitByAge.pdf')
        plt.show()
    
        plt.plot(Age,MyMarket.LogTotalMedStdByAge)
        plt.plot(Age,MyMarket.data_moments[100:160],'.k')
        plt.xlabel('Age')
        plt.ylabel('Stdev log total (nonzero) medical expenses')
        plt.xlim((25,85))
        #plt.savefig('../Figures/StdevTotalMedFitByAge.pdf')
        plt.show()
        
        # Make a "detrender" based on quadratic fit of data moments
        f = lambda x : 5.14607229 + 0.04242741*x
        LogMedMeanAdj = np.mean(np.reshape(f(Age),(12,5)),axis=1)
            
        plt.plot(Age5year,MyMarket.LogTotalMedMeanByAgeHealth - np.tile(np.reshape(LogMedMeanAdj,(12,1)),(1,5)))
        temp = np.reshape(MyMarket.data_moments[320:380],(12,5))
        plt.plot(Age5year,temp[:,0] - LogMedMeanAdj,'.b')
        plt.plot(Age5year,temp[:,1] - LogMedMeanAdj,'.g')
        plt.plot(Age5year,temp[:,2] - LogMedMeanAdj,'.r')
        plt.plot(Age5year,temp[:,3] - LogMedMeanAdj,'.c')
        plt.plot(Age5year,temp[:,4] - LogMedMeanAdj,'.m')
        plt.xlabel('Age')
        plt.ylabel('Detrended mean log total (nonzero) medical expenses')
        plt.title('Medical expenses by age group and health status')
        plt.xlim((25,85))
        #plt.savefig('../Figures/MeanTotalMedFitByAgeHealth.pdf')
        plt.show()
        
        plt.plot(Age5year,MyMarket.LogTotalMedStdByAgeHealth)
        temp = np.reshape(MyMarket.data_moments[380:440],(12,5))
        plt.plot(Age5year,temp[:,0],'.b')
        plt.plot(Age5year,temp[:,1],'.g')
        plt.plot(Age5year,temp[:,2],'.r')
        plt.plot(Age5year,temp[:,3],'.c')
        plt.plot(Age5year,temp[:,4],'.m')
        plt.xlabel('Age')
        plt.ylabel('Stdev log total (nonzero) medical expenses')
        plt.title('Medical expenses by age group and health status')
        plt.xlim((25,85))
        #plt.savefig('../Figures/StdevTotalMedFitByAgeHealth.pdf')
        plt.show()
            
        plt.plot(Age5year[:8],MyMarket.LogTotalMedMeanByAgeIncome - np.tile(np.reshape(LogMedMeanAdj[:8],(8,1)),(1,5)))
        temp = np.reshape(MyMarket.data_moments[480:520],(8,5))
        plt.plot(Age5year[:8],temp[:,0] - LogMedMeanAdj[:8],'.b')
        plt.plot(Age5year[:8],temp[:,1] - LogMedMeanAdj[:8],'.g')
        plt.plot(Age5year[:8],temp[:,2] - LogMedMeanAdj[:8],'.r')
        plt.plot(Age5year[:8],temp[:,3] - LogMedMeanAdj[:8],'.c')
        plt.plot(Age5year[:8],temp[:,4] - LogMedMeanAdj[:8],'.m')
        plt.xlabel('Age')
        plt.ylabel('Detrended mean log total (nonzero) medical expenses')
        plt.title('Medical expenses by age group and income quintile')
        plt.xlim((25,65))
        #plt.savefig('../Figures/MeanTotalMedFitByAgeIncome.pdf')
        plt.show()
        
        plt.plot(Age5year[:8],MyMarket.LogTotalMedStdByAgeIncome)
        temp = np.reshape(MyMarket.data_moments[520:560],(8,5))
        plt.plot(Age5year[:8],temp[:,0],'.b')
        plt.plot(Age5year[:8],temp[:,1],'.g')
        plt.plot(Age5year[:8],temp[:,2],'.r')
        plt.plot(Age5year[:8],temp[:,3],'.c')
        plt.plot(Age5year[:8],temp[:,4],'.m')
        plt.xlabel('Age')
        plt.ylabel('Stdev log total (nonzero) medical expenses')
        plt.title('Medical expenses by age group and income quintile')
        plt.xlim((25,65))
        #plt.savefig('../Figures/StdevTotalMedFitByAgeIncome.pdf')
        plt.show()
        
        plt.plot(Age[0:40],MyMarket.InsuredRateByAge_ESI,'-b')
        plt.plot(Age[0:40],MyMarket.data_moments[160:200],'.k')
        plt.xlabel('Age')
        plt.ylabel('ESI uptake rate')
        plt.xlim((25,65))
        #plt.savefig('../Figures/ESIuptakeFitByAge.pdf')
        plt.show()
        
        plt.plot(Age5year[:8],MyMarket.InsuredRateByAgeIncome)
        temp = np.reshape(MyMarket.data_moments[560:600],(8,5))
        plt.plot(Age5year[:8],temp[:,0],'.b')
        plt.plot(Age5year[:8],temp[:,1],'.g')
        plt.plot(Age5year[:8],temp[:,2],'.r')
        plt.plot(Age5year[:8],temp[:,3],'.c')
        plt.plot(Age5year[:8],temp[:,4],'.m')
        plt.xlabel('Age')
        plt.ylabel('ESI uptake rate')
        plt.xlim((25,65))
        #plt.savefig('../Figures/ESIuptakeFitByAgeIncome.pdf')
        plt.show()
        
        plt.plot(Age[0:40],MyMarket.PremiumMeanByAge,'-b')
        plt.plot(Age[0:40],MyMarket.data_moments[240:280],'.k')
        plt.xlabel('Age')
        plt.ylabel('Mean out-of-pocket premiums paid')
        plt.xlim((25,65))
        #plt.savefig('../Figures/OOPpremiumFitByAge.pdf')
        plt.show()
    #    
    #    plt.plot(Age[0:40],MyMarket.PremiumStdByAge,'-b')
    #    plt.plot(Age[0:40],MyMarket.data_moments[280:320],'.k')
    #    plt.xlabel('Age')
    #    plt.ylabel('Stdev out-of-pocket premiums paid')
    #    plt.xlim((25,65))
    #    plt.show()
    #    
        plt.plot(Age[0:40],MyMarket.ZeroSubsidyRateByAge,'-b')
        plt.plot(Age[0:40],MyMarket.data_moments[200:240],'.k')
        plt.xlabel('Age')
        plt.ylabel('Pct insured with zero employer contribution')
        plt.xlim((25,65))
        #plt.savefig('../Figures/ZeroEmpContrShareFitByAge.pdf')
        plt.show()
        
        plt.plot(Age,MyMarket.OOPshareByAge,'-b')
        plt.plot(Age,MyMarket.data_moments[760:820],'.k')
        plt.xlabel('Age')
        plt.ylabel('Share of medical expenses paid out-of-pocket')
        plt.xlim((25,85))
        #plt.savefig('../Figures/OOPshareFitByAge.pdf')
        plt.show()
    
     
    if test_static_model:
        #This block of code is for testing the static model
        t_start = clock()
        InsChoice = False
        MyMarket = makeMarketFromParams(Params.test_param_vec,np.array([1,2,3,4,5]),InsChoice)
        StaticType = MyMarket.agents[1]
        StaticType.update()
        StaticType.makeShockHistory()
        t_end = clock()
        print('Making a static agent type took ' + mystr(t_end-t_start) + ' seconds.')
        t_start = clock()
        StaticType.solve()
        t_end = clock()
        print('Solving a static agent type took ' + mystr(t_end-t_start) + ' seconds.')
        t_start = clock()
        StaticType.initializeSim()
        StaticType.simulate()
        t_end = clock()
        print('Simulating a static agent type took ' + mystr(t_end-t_start) + ' seconds.')


    if test_one_type:
        t_start = clock()
        # This block of code is for testing one type of agent
        EvalType  = 0  # Number of times to do a static search for eqbm premiums
        InsChoice = 1  # Extent of insurance choice
        TestPremiums = True # Whether to start with the test premium level
        
        if TestPremiums:
            ESIpremiums = np.array([0.3500, 0.0, 0.0, 0.0, 0.0])
        else:
            ESIpremiums = Params.PremiumsLast
        IMIpremiums_init = Params.IMIpremiums
        
        ContractCounts = [0,1,5] # plus one
        ESIpremiums_init_short = np.concatenate((np.array([0.]),ESIpremiums[0:ContractCounts[InsChoice]]))
        ESIpremiums_init = np.tile(np.reshape(ESIpremiums_init_short,(1,ESIpremiums_init_short.size)),(40,1))
        
        MyMarket = makeMarketFromParams(Params.test_param_vec,ageHealthRatedActuarialRule,IMIpremiums_init,ESIpremiums_init,InsChoice)
        MyMarket.HealthGroups = [[0],[1,2,3,4]]
        MyMarket.ExcludedGroups = [True,False]
        MyMarket.ESIpremiums = ESIpremiums_init_short
        multiThreadCommandsFake(MyMarket.agents,['update()','makeShockHistory()'])
        MyMarket.getIncomeQuintiles()
        multiThreadCommandsFake(MyMarket.agents,['makeIncBoolArray()'])
        t_end = clock()
        print('Making the agents took ' + mystr(t_end-t_start) + ' seconds.')
        
        t_start = clock()
        MyType = MyMarket.agents[0]
        MyType.del_soln = False
        MyType.do_sim = False
        MyType.verbosity = 10
        MyType.solve()
        t_end = clock()
        print('Solving and simulating one agent type took ' + str(t_end-t_start) + ' seconds.')
           
        t = 0
        p = 3.0    
        h = 9        
        Dev = 0.0
        z = 1
        mTop = 10.
        
        print('Individual market:')
        MyType.plotvFunc(t,p,H=[0,1,2,3,4],decurve=False,mMax=mTop)
        MyType.plotvPfunc(t,p,H=[0,1,2,3,4],decurve=False,mMax=mTop)
        if t < 40:
            print('ESI paying full price:')
            MyType.plotvFunc(t,p,H=[5,6,7,8,9],decurve=False,mMax=mTop)
            MyType.plotvPfunc(t,p,H=[5,6,7,8,9],decurve=False,mMax=mTop)
            print('ESI with employer contribution:')
            MyType.plotvFunc(t,p,H=[10,11,12,13,14],decurve=False,mMax=mTop)
            MyType.plotvPfunc(t,p,H=[10,11,12,13,14],decurve=False,mMax=mTop)
        
        MyType.plotvFuncByContract(t,h,p,mMax=mTop)
        MyType.plotcFuncByContract(t,h,p,Dev,mMax=mTop)
        MyType.plotcFuncByDev(t,h,z,p,mMax=mTop)
        MyType.plotMedFuncByDev(t,h,z,p,mMax=mTop)
        MyType.plotxFuncByDev(t,h,z,p,mMax=mTop)
        MyType.plotAVfuncByContract(t,h,p,mMax=mTop)
