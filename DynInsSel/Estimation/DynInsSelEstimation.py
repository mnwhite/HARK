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
from InsuranceSelectionModel import MedInsuranceContract, InsSelConsumerType
import SaveParameters
from SaveParameters import writeParametersToFile
import LoadDataMoments as Data
from LoadDataMoments import data_moments, moment_weights
from ActuarialRules import BaselinePolicySpec, InsuranceMarket
from HARKinterpolation import ConstantFunction
from HARKutilities import getPercentiles
from HARKparallel import multiThreadCommands, multiThreadCommandsFake
from joblib import Parallel, delayed
import dill as pickle


class DynInsSelType(InsSelConsumerType):
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
        Also computes net health spending by age.
        '''
        self.OOPmedHist = self.OOPnow_hist
        self.OOPmedHist[np.logical_and(self.OOPmedHist < 0.0001,self.OOPmedHist > 0.0,)] = 0.0001
        MedPrice_temp = np.tile(np.reshape(self.MedPrice[0:self.T_sim],(self.T_sim,1)),(1,self.AgentCount))
        self.TotalMedHist = self.MedLvlNow_hist*MedPrice_temp
        self.TotalMedHist[np.logical_and(self.TotalMedHist < 0.0001,self.TotalMedHist > 0.0,)] = 0.0001
        self.WealthRatioHist = self.aLvlNow_hist/self.pLvlHist
        self.InsuredBoolArray = self.ContractNow_hist > 0
        self.HealthBudgetByAge = np.sum(self.BudgetNow_hist,axis=1)
        
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
        HealthCount = 5
        for t in range(T):
            for h in range(HealthCount):
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
                these = self.MrkvHist[t,:] == h
                m = self.mLvlNow_hist[t,these]
                vTarg = self.vTarg_hist[t,these]
                vFunc = self.vFunc[t][h]
                pMin = vFunc.func.func.y_list[0] # Lowest pLvl
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
        
        # Initialize simulated moments by age
        MeanLogOOPmedByAge = np.zeros(60)   # 0:60
        MeanLogTotalMedByAge = np.zeros(60) # 60:120
        StdevLogOOPmedByAge = np.zeros(60)  # 120:180
        StdevLogTotalMedByAge = np.zeros(60)# 180:240
        OOPshareByAge = np.zeros(60)        # 240:300
        ESIinsuredRateByAge = np.zeros(40)  # 300:340
        IMIinsuredRateByAge = np.zeros(40)  # 340:380
        MeanESIpremiumByAge = np.zeros(40)  # 380:420
        StdevESIpremiumByAge = np.zeros(40) # 420:460
        NoPremShareRateByAge = np.zeros(40) # 460:500
        MedianWealthRatioByAge = np.zeros(40)# 1400:1440
        ESIofferRateByAge = np.zeros(40) # Can't be estimated, exogenous process

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
                those = np.logical_and(these,ThisType.OOPmedHist[t,:]>0.0)
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
                MedianWealthRatioByAge[t] = np.median(WealthRatioArray)
                PremiumArray = np.hstack(PremiumList)
                MeanESIpremiumByAge[t] = np.mean(PremiumArray)
                StdevESIpremiumByAge[t] = np.std(PremiumArray)
                ESIinsuredRateByAge[t] = ESIcount/OfferedCount
                IMIinsuredRateByAge[t] = IMIcount/NotOfferedCount
                ESIofferRateByAge[t] = OfferedCount/LiveCount
                if ESIcount > 0.0:
                    NoPremShareRateByAge[t] = ZeroCount/ESIcount
                else:
                    NoPremShareRateByAge[t] = 0.0 # This only happens with no insurance choice
            LogTotalMedArray = np.hstack(LogTotalMedList)
            MeanLogTotalMedByAge[t] = np.mean(LogTotalMedArray)
            StdevLogTotalMedByAge[t] = np.std(LogTotalMedArray)
            LogOOPmedArray = np.hstack(LogOOPmedList)
            MeanLogOOPmedByAge[t] = np.mean(LogOOPmedArray)
            StdevLogOOPmedByAge[t] = np.std(LogOOPmedArray)
            OOPshareByAge[t] = OOPmedSum/TotalMedSum
            
        # Initialize moments by age-health and define age band bounds
        MeanLogOOPmedByAgeHealth = np.zeros((12,5))   # 500:560
        MeanLogTotalMedByAgeHealth = np.zeros((12,5)) # 560:620
        StdevLogOOPmedByAgeHealth = np.zeros((12,5))  # 620:680
        StdevLogTotalMedByAgeHealth = np.zeros((12,5))# 680:740
        OOPshareByAgeHealth = np.zeros((12,5))        # 740:800
        ESIinsuredRateByAgeHealth = np.zeros((8,5))   # 800:840
        IMIinsuredRateByAgeHealth = np.zeros((8,5))   # 840:880
        MeanESIpremiumByAgeHealth = np.zeros((8,5))   # 880:920
        StdevESIpremiumByAgeHealth = np.zeros((8,5))  # 920:960
        NoPremShareRateByAgeHealth = np.zeros((8,5))  # 960:1000
        MedianWealthRatioByAgeHealth = np.zeros((8,5))# NO DATA
        ESIofferRateByAgeHealth = np.zeros((8,5)) # Can't be estimated, exogenous process
        AgeBounds = [[0,5],[5,10],[10,15],[15,20],[20,25],[25,30],[30,35],[35,40],[40,45],[45,50],[50,55],[55,60]]

        # Calculate all simulated moments by age-health
        for a in range(12):
            bot = AgeBounds[a][0]
            top = AgeBounds[a][1]
            for h in range(5):
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
                    these = ThisType.HealthBoolArray[bot:top,:,h]
                    those = np.logical_and(these,ThisType.TotalMedHist[bot:top,:]>0.0)
                    LogTotalMedList.append(np.log(ThisType.TotalMedHist[bot:top,:][those]))
                    those = np.logical_and(these,ThisType.OOPmedHist[bot:top,:]>0.0)
                    LogOOPmedList.append(np.log(ThisType.OOPmedHist[bot:top,:][those]))
                    
                    if a < 8:
                        WealthRatioList.append(ThisType.WealthRatioHist[bot:top,:][these])
                        Insured = ThisType.InsuredBoolArray[bot:top,:][these]
                        LiveCount += np.sum(ThisType.LiveBoolArray[bot:top,:][these])
                        MrkvTemp = ThisType.MrkvHist[bot:top,:][these]
                        Offered = MrkvTemp >= 5
                        NotOffered = np.logical_and(MrkvTemp < 5, MrkvTemp >= 0) 
                        HaveESI = np.logical_and(Insured, Offered)
                        ESIcount += np.sum(HaveESI)
                        OfferedCount += np.sum(Offered)
                        IMIcount += np.sum(np.logical_and(Insured,NotOffered))
                        NotOfferedCount += np.sum(NotOffered)
                        ZeroCount += np.sum(np.logical_and(HaveESI, MrkvTemp <= 9))
                        PremiumList.append(ThisType.PremNow_hist[bot:top,:][these][HaveESI])
                        TotalMedSum += np.sum(ThisType.TotalMedHist[bot:top,:][these][HaveESI])
                        OOPmedSum += np.sum(ThisType.OOPmedHist[bot:top,:][these][HaveESI])
                    else:
                        TotalMedSum += np.sum(ThisType.TotalMedHist[bot:top,:][these])
                        OOPmedSum += np.sum(ThisType.OOPmedHist[bot:top,:][these])
                        
                if a < 8:
                    WealthRatioArray = np.hstack(WealthRatioList)
                    MedianWealthRatioByAgeHealth[a,h] = np.median(WealthRatioArray)
                    PremiumArray = np.hstack(PremiumList)
                    MeanESIpremiumByAgeHealth[a,h] = np.mean(PremiumArray)
                    StdevESIpremiumByAgeHealth[a,h] = np.std(PremiumArray)
                    ESIinsuredRateByAgeHealth[a,h] = ESIcount/OfferedCount
                    IMIinsuredRateByAgeHealth[a,h] = IMIcount/NotOfferedCount
                    ESIofferRateByAgeHealth[a,h] = OfferedCount/LiveCount
                    if ESIcount > 0.0:
                        NoPremShareRateByAgeHealth[a,h] = ZeroCount/ESIcount
                    else:
                        NoPremShareRateByAgeHealth[a,h] = 0.0 # This only happens with no insurance choice
                    
                LogTotalMedArray = np.hstack(LogTotalMedList)
                MeanLogTotalMedByAgeHealth[a,h] = np.mean(LogTotalMedArray)
                StdevLogTotalMedByAgeHealth[a,h] = np.std(LogTotalMedArray)
                LogOOPmedArray = np.hstack(LogOOPmedList)
                MeanLogOOPmedByAgeHealth[a,h] = np.mean(LogOOPmedArray)
                StdevLogOOPmedByAgeHealth[a,h] = np.std(LogOOPmedArray)
                OOPshareByAgeHealth[a,h] = OOPmedSum/TotalMedSum
                
        # Initialize moments by age-income
        MeanLogOOPmedByAgeIncome = np.zeros((8,5))    # 1000:1040
        MeanLogTotalMedByAgeIncome = np.zeros((8,5))  # 1040:1080
        StdevLogOOPmedByAgeIncome = np.zeros((8,5))   # 1080:1120
        StdevLogTotalMedByAgeIncome = np.zeros((8,5)) # 1120:1160
        OOPshareByAgeIncome = np.zeros((8,5))         # 1160:1200
        ESIinsuredRateByAgeIncome = np.zeros((8,5))   # 1200:1240
        IMIinsuredRateByAgeIncome = np.zeros((8,5))   # 1240:1280
        MeanESIpremiumByAgeIncome = np.zeros((8,5))   # 1280:1320
        StdevESIpremiumByAgeIncome = np.zeros((8,5))  # 1320:1360
        NoPremShareRateByAgeIncome = np.zeros((8,5))  # 1360:1400
        MedianWealthRatioByAgeIncome = np.zeros((8,5))# 1440:1480
        ESIofferRateByAgeIncome = np.zeros((8,5)) # Can't be estimated, exogenous process

        # Calculated all simulated moments by age-income
        for a in range(8):
            bot = AgeBounds[a][0]
            top = AgeBounds[a][1]
            for i in range(5):
                WealthRatioList = []
                LogTotalMedList = []
                LogOOPmedList = []
                PremiumList = []
                TotalMedSum = 0.0
                OOPmedSum = 0.0
                LiveCount = 0.0
                ESIcount = 0.0
                IMIcount = 0.0
                OfferedCount = 0.0
                NotOfferedCount = 0.0
                ZeroCount = 0.0
                
                for ThisType in Agents:
                    these = ThisType.IncQuintBoolArray[bot:top,:,i]
                    those = np.logical_and(these,ThisType.TotalMedHist[bot:top,:] > 0.0)
                    LogTotalMedList.append(np.log(ThisType.TotalMedHist[bot:top,:][those]))
                    those = np.logical_and(these,ThisType.OOPmedHist[bot:top,:] > 0.0)
                    LogOOPmedList.append(np.log(ThisType.OOPmedHist[bot:top,:][those]))
                    
                    LiveCount += np.sum(ThisType.LiveBoolArray[bot:top,:][these])
                    Insured = ThisType.InsuredBoolArray[bot:top,:][these]
                    MrkvTemp = ThisType.MrkvHist[bot:top,:][these]
                    Offered = MrkvTemp >= 5
                    OfferedCount += np.sum(Offered)
                    NotOffered = np.logical_and(MrkvTemp < 5, MrkvTemp >= 0) 
                    NotOfferedCount += np.sum(NotOffered)
                    WealthRatioList.append(ThisType.WealthRatioHist[bot:top,:][these])
                    HaveESI = np.logical_and(Insured,Offered)
                    ESIcount += np.sum(HaveESI)
                    HaveIMI = np.logical_and(Insured,NotOffered)
                    IMIcount += np.sum(HaveIMI)
                    ZeroCount += np.sum(np.logical_and(HaveESI, MrkvTemp <= 9))
                    PremiumList.append(ThisType.PremNow_hist[bot:top,][these][HaveESI])
                    TotalMedSum += np.sum(ThisType.TotalMedHist[bot:top,:][these][HaveESI])
                    OOPmedSum += np.sum(ThisType.OOPmedHist[bot:top,:][these][HaveESI])
                WealthRatioArray = np.hstack(WealthRatioList)
                MedianWealthRatioByAgeIncome[a,i] = np.median(WealthRatioArray)
                PremiumArray = np.hstack(PremiumList)
                MeanESIpremiumByAgeIncome[a,i] = np.mean(PremiumArray)
                StdevESIpremiumByAgeIncome[a,i] = np.std(PremiumArray)
                ESIinsuredRateByAgeIncome[a,i] = ESIcount/OfferedCount
                IMIinsuredRateByAgeIncome[a,i] = IMIcount/NotOfferedCount
                ESIofferRateByAgeIncome[a,i] = OfferedCount/LiveCount
                LogTotalMedArray = np.hstack(LogTotalMedList)
                MeanLogTotalMedByAgeIncome[a,i] = np.mean(LogTotalMedArray)
                StdevLogTotalMedByAgeIncome[a,i] = np.std(LogTotalMedArray)
                LogOOPmedArray = np.hstack(LogOOPmedList)
                MeanLogOOPmedByAgeIncome[a,i] = np.mean(LogOOPmedArray)
                StdevLogOOPmedByAgeIncome[a,i] = np.std(LogOOPmedArray)
                OOPshareByAgeIncome[a,i] = OOPmedSum/TotalMedSum
                if ESIcount > 0.0:
                    NoPremShareRateByAgeIncome[a,i] = ZeroCount/ESIcount
                else:
                    NoPremShareRateByAgeIncome[a,i] = 0.0 # This only happens with no insurance choice
                    
        
        # Store all of the simulated moments as attributes of self
        self.MeanLogOOPmedByAge = MeanLogOOPmedByAge + np.log(10000)
        self.MeanLogTotalMedByAge = MeanLogTotalMedByAge + np.log(10000)
        self.StdevLogOOPmedByAge = StdevLogOOPmedByAge
        self.StdevLogTotalMedByAge = StdevLogTotalMedByAge
        self.OOPshareByAge = OOPshareByAge
        self.ESIinsuredRateByAge = ESIinsuredRateByAge
        self.IMIinsuredRateByAge = IMIinsuredRateByAge
        self.MeanESIpremiumByAge = MeanESIpremiumByAge*10000
        self.StdevESIpremiumByAge = StdevESIpremiumByAge*10000
        self.NoPremShareRateByAge = NoPremShareRateByAge
        self.MedianWealthRatioByAge = MedianWealthRatioByAge
        self.MeanLogOOPmedByAgeHealth = MeanLogOOPmedByAgeHealth + np.log(10000)
        self.MeanLogTotalMedByAgeHealth = MeanLogTotalMedByAgeHealth + np.log(10000)
        self.StdevLogOOPmedByAgeHealth = StdevLogOOPmedByAgeHealth
        self.StdevLogTotalMedByAgeHealth = StdevLogTotalMedByAgeHealth
        self.OOPshareByAgeHealth = OOPshareByAgeHealth
        self.ESIinsuredRateByAgeHealth = ESIinsuredRateByAgeHealth
        self.IMIinsuredRateByAgeHealth = IMIinsuredRateByAgeHealth
        self.MeanESIpremiumByAgeHealth = MeanESIpremiumByAgeHealth*10000
        self.StdevESIpremiumByAgeHealth = StdevESIpremiumByAgeHealth*10000
        self.NoPremShareRateByAgeHealth = NoPremShareRateByAgeHealth
        self.MedianWealthRatioByAgeHealth = MedianWealthRatioByAgeHealth
        self.MeanLogOOPmedByAgeIncome = MeanLogOOPmedByAgeIncome + np.log(10000)
        self.MeanLogTotalMedByAgeIncome = MeanLogTotalMedByAgeIncome + np.log(10000)
        self.StdevLogOOPmedByAgeIncome = StdevLogOOPmedByAgeIncome
        self.StdevLogTotalMedByAgeIncome = StdevLogTotalMedByAgeIncome
        self.OOPshareByAgeIncome = OOPshareByAgeIncome
        self.ESIinsuredRateByAgeIncome = ESIinsuredRateByAgeIncome
        self.IMIinsuredRateByAgeIncome = IMIinsuredRateByAgeIncome
        self.MeanESIpremiumByAgeIncome = MeanESIpremiumByAgeIncome*10000
        self.StdevESIpremiumByAgeIncome = StdevESIpremiumByAgeIncome*10000
        self.NoPremShareRateByAgeIncome = NoPremShareRateByAgeIncome
        self.MedianWealthRatioByAgeIncome = MedianWealthRatioByAgeIncome
        self.ESIofferRateByAge = ESIofferRateByAge
        self.ESIofferRateByAgeHealth = ESIofferRateByAgeHealth
        self.ESIofferRateByAgeIncome = ESIofferRateByAgeIncome
        
        
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
        MomentList = [
            self.MeanLogOOPmedByAge,
            self.MeanLogTotalMedByAge,
            self.StdevLogOOPmedByAge,
            self.StdevLogTotalMedByAge,
            self.OOPshareByAge,
            self.ESIinsuredRateByAge,
            self.IMIinsuredRateByAge,
            self.MeanESIpremiumByAge,
            self.StdevESIpremiumByAge,
            self.NoPremShareRateByAge,
            self.MeanLogOOPmedByAgeHealth.flatten(),
            self.MeanLogTotalMedByAgeHealth.flatten(),
            self.StdevLogOOPmedByAgeHealth.flatten(),
            self.StdevLogTotalMedByAgeHealth.flatten(),
            self.OOPshareByAgeHealth.flatten(),
            self.ESIinsuredRateByAgeHealth.flatten(),
            self.IMIinsuredRateByAgeHealth.flatten(),
            self.MeanESIpremiumByAgeHealth.flatten(),
            self.StdevESIpremiumByAgeHealth.flatten(),
            self.NoPremShareRateByAgeHealth.flatten(),
            self.MeanLogOOPmedByAgeIncome.flatten(),
            self.MeanLogTotalMedByAgeIncome.flatten(),
            self.StdevLogOOPmedByAgeIncome.flatten(),
            self.StdevLogTotalMedByAgeIncome.flatten(),
            self.OOPshareByAgeIncome.flatten(),
            self.ESIinsuredRateByAgeIncome.flatten(),
            self.IMIinsuredRateByAgeIncome.flatten(),
            self.MeanESIpremiumByAgeIncome.flatten(),
            self.StdevESIpremiumByAgeIncome.flatten(),
            self.NoPremShareRateByAgeIncome.flatten(),
            self.MedianWealthRatioByAge,
            self.MedianWealthRatioByAgeIncome.flatten()
            ]    
        self.sim_moments = np.concatenate(MomentList)

        
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
        moment_diff = self.sim_moments - self.data_moments
        moment_diff[self.moment_weights == 0.0] = 0.0 # Just in case
        moment_sum = np.dot(moment_diff**2,self.moment_weights)
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
                      EmpContr,EducType,InsChoiceType,ContractCount):
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
    EmpContr : float
        Employer contribution to any (non-null) ESI contract (when in Mrkv>=10).
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
    
    if InsChoiceType > 0:
        IndMarketContractList = [MedInsuranceContract(ConstantFunction(0.0),0.0,1.0,Params.MedPrice),
                                 MedInsuranceContract(ConstantFunction(0.0),0.2,Copay,Params.MedPrice)]
    else:
         IndMarketContractList = [MedInsuranceContract(ConstantFunction(0.0),0.2,0.08,Params.MedPrice)]   
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
    ThisType.track_vars = ['aLvlNow','mLvlNow','BudgetNow','cLvlNow','MedLvlNow','PremNow','ContractNow','OOPnow']
    ThisType.EmpContr = EmpContr
    if EmpContr == 0.0:
        ThisType.ZeroSubsidyBool = True
    else:
        ThisType.ZeroSubsidyBool = False
    return ThisType
        

def makeMarketFromParams(ParamArray,PolicySpec,IMIpremiumArray,ESIpremiumArray,InsChoiceType):
    '''
    Makes a list of DynInsSelTypes, to be used for estimation.
    
    Parameters
    ----------
    ParamArray : np.array
        Array of size 35, representing all of the structural parameters.
    PolicySpec : PolicySpecification
        Object describing the policy structure of the insurance market.
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
        ThisAgent = makeDynInsSelType(CRRAcon,MedCurve,DiscFac,BequestShift,BequestScale,Cfloor,ChoiceShkMag,
                  MedShkMeanAgeParams,MedShkMeanVGparams,MedShkMeanGDparams,MedShkMeanFRparams,MedShkMeanPRparams,
                  MedShkStdAgeParams,MedShkStdVGparams,MedShkStdGDparams,MedShkStdFRparams,
                  MedShkStdPRparams,EmpContr,k,InsChoiceType,ContractCount)
        ThisAgent.Weight = Params.EducWeight[k]
        ThisAgent.AgentCount = int(round(ThisAgent.Weight*Params.AgentCountTotal))
        AgentList.append(ThisAgent)
        i += 1
    
    for i in range(len(AgentList)):
        AgentList[i].seed = i # Assign different seeds to each type
        
    # Make an expanded array of IMI premiums
    GroupCount = len(PolicySpec.HealthGroups)
    IMIpremiumArray_big = np.zeros((5,40))
    for g in range(GroupCount):
        for h in PolicySpec.HealthGroups[g]:
            if PolicySpec.ExcludedGroups[g]:
                IMIpremiumArray_big[h,:] = 10000.
            else:
                IMIpremiumArray_big[h,:] = IMIpremiumArray[g,:]
            
    # Construct an initial nested list for premiums
    PremiumFuncs_init = []
    HealthCount = 5
    for t in range(40):
        PremiumFuncs_t = []
        for h in range(HealthCount):
            PremiumFuncs_this_health = []
            for z in range(ContractCount):
                if z > 0:
                    Prem = IMIpremiumArray_big[h,t]
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
    ThisMarket = DynInsSelMarket(PolicySpec,AgentList)
    
    ThisMarket.data_moments = data_moments
    ThisMarket.moment_weights = moment_weights
    ThisMarket.PremiumFuncs_init = PremiumFuncs_init
    ThisMarket.LoadFacESI   = Params.LoadFacESI
    ThisMarket.LoadFacIMI   = Params.LoadFacIMI
    ThisMarket.CohortGroFac = Params.CohortGroFac
    
    # Have each agent type in the market inherit the premium functions
    for this_agent in ThisMarket.agents:
        setattr(this_agent,'PremiumFuncs',PremiumFuncs_init)
    
    #print('I made an insurance market with ' + str(len(InsuranceMarket.agents)) + ' agent types!')
    return ThisMarket


def objectiveFunction(Parameters, return_market=False):
    '''
    The objective function for the estimation.  Makes and solves a market, then
    returns the weighted sum of moment differences between simulation and data.
    '''
    EvalType  = 0  # Number of times to do a static search for eqbm premiums
    InsChoice = 1  # Extent of insurance choice
    TestPremiums = True # Whether to start with the test premium level
    
    if TestPremiums:
        ESIpremiums = np.array([0.3000, 0.0, 0.0, 0.0, 0.0])
    else:
        ESIpremiums = Params.PremiumsLast
    IMIpremiums_init = Params.IMIpremiums
    
    ContractCounts = [0,1,5] # plus one
    ESIpremiums_init_short = np.concatenate((np.array([0.]),ESIpremiums[0:ContractCounts[InsChoice]]))
    ESIpremiums_init = np.tile(np.reshape(ESIpremiums_init_short,(1,ESIpremiums_init_short.size)),(40,1))
    
    MyMarket = makeMarketFromParams(Parameters,BaselinePolicySpec,IMIpremiums_init,ESIpremiums_init,InsChoice)
    MyMarket.ESIpremiums = ESIpremiums_init_short
    multiThreadCommandsFake(MyMarket.agents,['update()','makeShockHistory()'])
    MyMarket.getIncomeQuintiles()
    multiThreadCommandsFake(MyMarket.agents,['makeIncBoolArray()'])
    
    if EvalType == 0:
        multiThreadCommandsFake(MyMarket.agents,['solve()'])
    else:
        MyMarket.max_loops = EvalType
        MyMarket.solve()
        Params.PremiumsLast = MyMarket.ESIpremiums

    MyMarket.calcSimulatedMoments()
    MyMarket.combineSimulatedMoments()
    moment_sum = MyMarket.aggregateMomentConditions()
    writeParametersToFile(Parameters,'LatestParameters.txt')    
    
    #print(moment_sum)
    if return_market:
        return moment_sum, MyMarket
    else:
        return moment_sum


###############################################################################
    
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    mystr = lambda number : "{:.4f}".format(number)
    
    test_obj_func = False
    test_one_type = False
    perturb_one_param = True
    
    if test_obj_func:
    # This block is for actually testing the objective function
        t_start = clock()
        moment_sum, MyMarket = objectiveFunction(Params.test_param_vec, return_market=True)
        t_end = clock()
        print('Objective function evaluation took ' + mystr(t_end-t_start) + ' seconds, value is ' + str(moment_sum) + '.')
        
        # This block of code is for displaying moment fits after running objectiveFunc  
        Age = np.arange(25,85)
        Age5year = 27.5 + 5*np.arange(12)
        color_list = ['b','g','r','c','m']
        
        if not Params.StaticBool:
            plt.plot(Age[0:40],MyMarket.MedianWealthRatioByAge)
            plt.plot(Age[0:40],Data.MedianWealthRatioByAge,'.k')
            plt.xlabel('Age')
            plt.ylabel('Median wealth/income ratio')
            #plt.savefig('../Figures/WealthFitByAge.pdf')
            plt.show()
            
            sim_temp = MyMarket.MedianWealthRatioByAgeIncome
            data_temp = Data.MedianWealthRatioByAgeIncome
            for i in range(5):
                plt.plot(Age5year[:8],sim_temp[:,i],'-'+color_list[i])
            for i in range(5):
                plt.plot(Age5year[:8],data_temp[:,i],'.'+color_list[i])
            plt.xlabel('Age')
            plt.ylabel('Median wealth/income ratio by income quintile')
            #plt.savefig('../Figures/WealthFitByAgeIncome.pdf')
            plt.show()
        
        plt.plot(Age,MyMarket.MeanLogTotalMedByAge)
        plt.plot(Age,Data.MeanLogTotalMedByAge,'.k')
        plt.xlabel('Age')
        plt.ylabel('Mean log total (nonzero) medical expenses')
        plt.xlim((25,85))
        #plt.savefig('../Figures/MeanTotalMedFitByAge.pdf')
        plt.show()
    
        plt.plot(Age,MyMarket.StdevLogTotalMedByAge)
        plt.plot(Age,Data.StdevLogTotalMedByAge,'.k')
        plt.xlabel('Age')
        plt.ylabel('Stdev log total (nonzero) medical expenses')
        plt.xlim((25,85))
        #plt.savefig('../Figures/StdevTotalMedFitByAge.pdf')
        plt.show()
        
        # Make a "detrender" based on quadratic fit of data moments
        f = lambda x : 5.14607229 + 0.04242741*x
        LogMedMeanAdj = np.mean(np.reshape(f(Age),(12,5)),axis=1)
            
        sim_temp = MyMarket.MeanLogTotalMedByAgeHealth - np.tile(np.reshape(LogMedMeanAdj,(12,1)),(1,5))
        data_temp = Data.MeanLogTotalMedByAgeHealth - np.tile(np.reshape(LogMedMeanAdj,(12,1)),(1,5))
        for h in range(5):
            plt.plot(Age5year,sim_temp[:,h],'-'+color_list[h])
            plt.plot(Age5year,data_temp[:,h],'.'+color_list[h])
        plt.xlabel('Age')
        plt.ylabel('Detrended mean log total (nonzero) medical expenses')
        plt.title('Medical expenses by age group and health status')
        plt.xlim((25,85))
        #plt.savefig('../Figures/MeanTotalMedFitByAgeHealth.pdf')
        plt.show()
        
        sim_temp = MyMarket.StdevLogTotalMedByAgeHealth
        data_temp = Data.StdevLogTotalMedByAgeHealth
        for h in range(5):
            plt.plot(Age5year,sim_temp[:,h],'-'+color_list[h])
            plt.plot(Age5year,data_temp[:,h],'.'+color_list[h])
        plt.xlabel('Age')
        plt.ylabel('Stdev log total (nonzero) medical expenses')
        plt.title('Medical expenses by age group and health status')
        plt.xlim((25,85))
        #plt.savefig('../Figures/StdevTotalMedFitByAgeHealth.pdf')
        plt.show()
            
        sim_temp = MyMarket.MeanLogTotalMedByAgeIncome - np.tile(np.reshape(LogMedMeanAdj[:8],(8,1)),(1,5))
        data_temp = Data.MeanLogTotalMedByAgeIncome - np.tile(np.reshape(LogMedMeanAdj[:8],(8,1)),(1,5))
        for h in range(5):
            plt.plot(Age5year[:8],sim_temp[:,h],'-'+color_list[h])
            plt.plot(Age5year[:8],data_temp[:,h],'.'+color_list[h])
        plt.xlabel('Age')
        plt.ylabel('Detrended mean log total (nonzero) medical expenses')
        plt.title('Medical expenses by age group and income quintile')
        plt.xlim((25,65))
        #plt.savefig('../Figures/MeanTotalMedFitByAgeIncome.pdf')
        plt.show()
        
        sim_temp = MyMarket.StdevLogTotalMedByAgeIncome
        data_temp = Data.StdevLogTotalMedByAgeIncome
        for h in range(5):
            plt.plot(Age5year[:8],sim_temp[:,h],'-'+color_list[h])
            plt.plot(Age5year[:8],data_temp[:,h],'.'+color_list[h])
        plt.xlabel('Age')
        plt.ylabel('Stdev log total (nonzero) medical expenses')
        plt.title('Medical expenses by age group and income quintile')
        plt.xlim((25,65))
        #plt.savefig('../Figures/StdevTotalMedFitByAgeIncome.pdf')
        plt.show()
        
        plt.plot(Age[0:40],MyMarket.ESIinsuredRateByAge,'-b')
        plt.plot(Age[0:40],Data.ESIinsuredRateByAge,'.k')
        plt.xlabel('Age')
        plt.ylabel('Employer sponsored insurance uptake rate')
        plt.xlim((25,65))
        #plt.savefig('../Figures/ESIuptakeFitByAge.pdf')
        plt.show()
        
        sim_temp = MyMarket.ESIinsuredRateByAgeIncome
        data_temp = Data.ESIinsuredRateByAgeIncome
        for h in range(5):
            plt.plot(Age5year[:8],sim_temp[:,h],'-'+color_list[h])
            plt.plot(Age5year[:8],data_temp[:,h],'.'+color_list[h])
        plt.xlabel('Age')
        plt.ylabel('ESI uptake rate')
        plt.xlim((25,65))
        #plt.savefig('../Figures/ESIuptakeFitByAgeIncome.pdf')
        plt.show()
        
        plt.plot(Age[0:40],MyMarket.IMIinsuredRateByAge,'-b')
        plt.plot(Age[0:40],Data.IMIinsuredRateByAge,'.k')
        plt.xlabel('Age')
        plt.ylabel('Individual market insured rate')
        plt.xlim((25,65))
        #plt.savefig('../Figures/IMIuptakeFitByAge.pdf')
        plt.show()
        
        sim_temp = MyMarket.IMIinsuredRateByAgeIncome
        data_temp = Data.IMIinsuredRateByAgeIncome
        for h in range(5):
            plt.plot(Age5year[:8],sim_temp[:,h],'-'+color_list[h])
            plt.plot(Age5year[:8],data_temp[:,h],'.'+color_list[h])
        plt.xlabel('Age')
        plt.ylabel('IMI insured rate by income quintile')
        plt.xlim((25,65))
        #plt.savefig('../Figures/IMIuptakeFitByAgeIncome.pdf')
        plt.show()
        
        plt.plot(Age[0:40],MyMarket.MeanESIpremiumByAge,'-b')
        plt.plot(Age[0:40],Data.MeanESIpremiumByAge,'.k')
        plt.xlabel('Age')
        plt.ylabel('Mean out-of-pocket premiums paid')
        plt.xlim((25,65))
        #plt.savefig('../Figures/OOPpremiumFitByAge.pdf')
        plt.show()

        plt.plot(Age[0:40],MyMarket.NoPremShareRateByAge,'-b')
        plt.plot(Age[0:40],Data.NoPremShareRateByAge,'.k')
        plt.xlabel('Age')
        plt.ylabel('Pct insured with zero employer contribution')
        plt.xlim((25,65))
        #plt.savefig('../Figures/ZeroEmpContrShareFitByAge.pdf')
        plt.show()
        
        plt.plot(Age,MyMarket.OOPshareByAge,'-b')
        plt.plot(Age,Data.OOPshareByAge,'.k')
        plt.xlabel('Age')
        plt.ylabel('Share of medical expenses paid out-of-pocket')
        plt.xlim((25,85))
        #plt.savefig('../Figures/OOPshareFitByAge.pdf')
        plt.show()
    
     
    if test_one_type:
        # This block of code is for testing one type of agent
        
        t_start = clock()
        EvalType  = 0  # Number of times to do a static search for eqbm premiums
        InsChoice = 1  # Extent of insurance choice
        TestPremiums = True # Whether to start with the test premium level
        
        if TestPremiums:
            ESIpremiums = np.array([0.3000, 0.0, 0.0, 0.0, 0.0])
        else:
            ESIpremiums = Params.PremiumsLast
        IMIpremiums_init = Params.IMIpremiums
        
        ContractCounts = [0,1,5] # plus one
        ESIpremiums_init_short = np.concatenate((np.array([0.]),ESIpremiums[0:ContractCounts[InsChoice]]))
        ESIpremiums_init = np.tile(np.reshape(ESIpremiums_init_short,(1,ESIpremiums_init_short.size)),(40,1))
        
        MyMarket = makeMarketFromParams(Params.test_param_vec,BaselinePolicySpec,IMIpremiums_init,ESIpremiums_init,InsChoice)
        MyMarket.ESIpremiums = ESIpremiums_init_short
        multiThreadCommandsFake(MyMarket.agents,['update()','makeShockHistory()'])
        MyMarket.getIncomeQuintiles()
        multiThreadCommandsFake(MyMarket.agents,['makeIncBoolArray()'])
        t_end = clock()
        print('Making the agents took ' + mystr(t_end-t_start) + ' seconds.')
        
        t_start = clock()
        MyType = MyMarket.agents[2]
        MyType.del_soln = False
        MyType.do_sim = True
        MyType.verbosity = 10
        MyType.solve()
        t_end = clock()
        print('Solving and simulating one agent type took ' + str(t_end-t_start) + ' seconds.')
           
        t = 0
        p = 3.0    
        h = 9        
        Dev = 0.0
        z = 0
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
        
        
    if perturb_one_param:
        # Test model identification by perturbing one parameter at a time
        param_i = 4
        param_min = 0.12
        param_max = 0.24
        N = 35
        perturb_vec = np.linspace(param_min,param_max,num=N)
        
        t_start = clock()
        parameter_set_list = []
        for j in range(N):
            params = copy(Params.test_param_vec)
            params[param_i] = perturb_vec[j]
            parameter_set_list.append(params)
#            fit_vec[j] = objectiveFunction(params)
        
        fit_list = Parallel(n_jobs=7)(delayed(objectiveFunction)(params) for params in parameter_set_list)
        fit_vec = np.array(fit_list)
        
        t_end = clock()
        print('Evaluating the objective function ' + str(N) + ' times took ' + str(t_end-t_start) + ' seconds.')
            
        plt.plot(perturb_vec,fit_vec)
        plt.xlabel(SaveParameters.param_names[param_i])
        plt.ylabel('Sum of squared moment differences')
        plt.savefig('../Figures/Perturb' + SaveParameters.param_names[param_i] + '.pdf')
        plt.show()
