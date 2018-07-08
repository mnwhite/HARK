'''
This module runs the structural estimation for the Ounce of Prevention project.
'''

import sys
sys.path.insert(0,'../')

from time import clock
from copy import copy
import numpy as np
from scipy.optimize import brentq, newton
from statsmodels.api import WLS
from HARKparallel import multiThreadCommands, multiThreadCommandsFake
from HARKestimation import minimizeNelderMead
from HARKutilities import getPercentiles
from HealthInvModel import HealthInvestmentConsumerType
import LoadHealthInvData as Data
import HealthInvParams as Params
from MakeTables import makeParamTable
import matplotlib.pyplot as plt

# Import objects from the data loading module
DataMoments = Data.all_moments
MomentWeights = Data.weighting_matrix
MomentMask = Data.moment_mask
MomentCount = Data.moment_count

class EstimationAgentType(HealthInvestmentConsumerType):
    '''
    A very small modification of HealthInvestmentConsumerType that readies it
    for use in the ounce of prevention estimation.
    '''
    def estimationAction(self):
        '''
        Run some commands for the estimation.
        '''
        self.update()
        self.solve()
        self.repSimData()
        self.initializeSim()
        self.simulate()
        self.delSolution()
        
        
    def repSimData(self):
        '''
        Replicate the HRS data a given number of times, so that there are more
        simulated agents than data respondents.
        '''
        X = self.DataToSimRepFactor
        self.AgentCount = X*self.DataAgentCount
        self.WealthQuint = np.tile(self.WealthQuint,X)
        self.HealthTert = np.tile(self.HealthTert,X)
        self.HealthQuint = np.tile(self.HealthQuint,X)
        self.aLvlInit = np.tile(self.aLvlInit,X)
        self.HlvlInit = np.tile(self.HlvlInit,X)
        self.t_ageInit = np.tile(self.t_ageInit,X)
        self.BornBoolArray = np.tile(self.BornBoolArray,(1,X))
        self.InDataSpanArray = np.tile(self.InDataSpanArray,(1,X))
        if self.Sex:
            self.SexLong = np.ones(self.AgentCount,dtype=bool)
        else:
            self.SexLong = np.zeros(self.AgentCount,dtype=bool)
        self.IncQuintLong = self.IncQuint*np.ones(self.AgentCount)
    
        
    def delSolution(self):
        del self.solution
        self.delFromTimeVary('solution')
        del self.ConvexityFixer
        self.delFromTimeInv('ConvexityFixer')
        
        
    def updateHealthProdFuncs(self):
        '''
        Defines the time-invariant attributes HealthProdFunc, HealthProdInvFunc,
        MargHealthProdFunc, and MargHealthProdInvFunc.  Translates the primitive
        parameters LogJerk, LogSlope, and LogCurve.
        '''
        tempw = np.exp(self.LogJerk)
        HealthProd0 = 1. - tempw
        tempx = np.exp(self.LogSlope) # Slope of health production function at iLvl=0
        HealthProd2 = np.exp(self.LogJerk - self.LogCurve)
        HealthProd1 = tempx/HealthProd0*HealthProd2**(1.-HealthProd0)
        if tempx > 0.:
            HealthProdFunc = lambda i : tempx/HealthProd0*((i*HealthProd2**((1.-HealthProd0)/HealthProd0) + HealthProd2**(1./HealthProd0))**HealthProd0 - HealthProd2)
            MargHealthProdFunc = lambda i : tempx*(i/HealthProd2 + 1.)**(HealthProd0-1.)
            MargHealthProdInvFunc = lambda q : HealthProd2*((q/tempx)**(1./(HealthProd0-1.)) - 1.)
        else:
            HealthProdFunc = lambda i : 0.*i
            MargHealthProdInvFunc = lambda q : 0.*q
            MargHealthProdFunc = lambda i : 0.*i
        
        # Define the (marginal)(inverse) health production function
        self.HealthProdFunc = HealthProdFunc
        self.MargHealthProdInvFunc = MargHealthProdInvFunc
        self.MargHealthProdFunc = MargHealthProdFunc
        self.addToTimeInv('HealthProdFunc','MargHealthProdFunc','MargHealthProdInvFunc')
        self.HealthProd0 = HealthProd0
        self.HealthProd1 = HealthProd1
        self.HealthProd2 = HealthProd2


def makeMultiTypeWithCohorts(params):
    '''
    Create 150 instances of the estimation agent type by splitting respondents
    by cohort-sex-income quintile.  Passes the parameter dictionary to each one.
    
    Parameters
    ----------
    params : dict
        The dictionary to be used to construct each of the 150 types.
        
    Returns
    -------
    type_list : [EstimationAgentType]
        List of 150 agent types to solve and simulate.
    '''
    type_list = []
    for n in range(150):
        temp_dict = copy(params)
        temp_dict['Sex'] = np.mod(n,10) >= 5 # males are 5,6,7,8,9
        CohortNum = (n-1)/10
        ThisType = EstimationAgentType(**temp_dict)
        ThisType.IncomeNow = Data.IncomeArray[n,:].tolist()
        ThisType.IncomeNext = Data.IncomeArray[n,1:].tolist() + [1.]
        MedPricePath = Data.MedPriceHistory[CohortNum:(CohortNum+ThisType.T_cycle)].tolist()
        ThisType.addToTimeVary('IncomeNow','IncomeNext','MedPrice')
        ThisType.MedPrice = MedPricePath
        ThisType.CohortNum = CohortNum + 1
        ThisType.IncQuint = np.mod(n,5)+1
        
        these = Data.TypeBoolArray[n,:]
        ThisType.DataAgentCount = np.sum(these)
        ThisType.WealthQuint = Data.wealth_quint_data[these]
        ThisType.HealthTert = Data.health_tert_data[these]
        ThisType.HealthQuint = Data.health_quint_data[these]
        ThisType.aLvlInit = Data.w_init[these]
        ThisType.HlvlInit = Data.h_init[these]
        ThisType.BornBoolArray = Data.BornBoolArray[:,these]
        ThisType.InDataSpanArray = Data.InDataSpanArray[:,these]
        ThisType.track_vars = ['OOPmedNow','hLvlNow','aLvlNow','CumLivPrb','DiePrbNow']
        ThisType.seed = n
        
        type_list.append(ThisType)
        
    return type_list


def makeMultiTypeSimple(params):
    '''
    Create 10 instances of the estimation agent type by splitting respondents
    by sex-income quintile.  Passes the parameter dictionary to each one.
    
    Parameters
    ----------
    params : dict
        The dictionary to be used to construct each of the 10 types.
        
    Returns
    -------
    type_list : [EstimationAgentType]
        List of 10 agent types to solve and simulate.
    '''
    type_list = []
    for n in range(10):
        temp_dict = copy(params)
        temp_dict['Sex'] = n >= 5 # males are 5,6,7,8,9
        ThisType = EstimationAgentType(**temp_dict)
        ThisType.IncomeNow = Data.IncomeArraySmall[n,:].tolist()
        ThisType.IncomeNext = Data.IncomeArraySmall[n,1:].tolist() + [1.]
        ThisType.addToTimeVary('IncomeNow','IncomeNext')
        ThisType.makeConstantMedPrice()
        ThisType.CohortNum = np.nan
        ThisType.IncQuint = np.mod(n,5)+1
        
        these = Data.TypeBoolArraySmall[n,:]
        ThisType.DataAgentCount = np.sum(these)
        ThisType.WealthQuint = Data.wealth_quint_data[these]
        ThisType.HealthTert = Data.health_tert_data[these]
        ThisType.HealthQuint = Data.health_quint_data[these]
        ThisType.aLvlInit = Data.w_init[these]
        ThisType.HlvlInit = Data.h_init[these]
        ThisType.BornBoolArray = Data.BornBoolArray[:,these]
        ThisType.t_ageInit = np.zeros_like(ThisType.aLvlInit)*np.nan # unused in estimation
        ThisType.InDataSpanArray = Data.InDataSpanArray[:,these]
        ThisType.track_vars = ['OOPmedNow','hLvlNow','aLvlNow','CumLivPrb','DiePrbNow','RatioNow','MedLvlNow','CopayMedNow','CopayInvstNow']
        ThisType.seed = n
        
        type_list.append(ThisType)
        
    return type_list


def processSimulatedTypes(params,use_cohorts):
    '''
    Make several types (150 or 10), and solve and simulate all of them.
    Returns the list of agents with the solution deleted (simulation results only).
    
    Parameters
    ----------
    params : dict
        The dictionary to be used to construct each of the types.
    use_cohorts : bool
        Indicator for whether to separately solve and simulate the 15 cohorts.
        
    Returns
    -------
    type_list : [EstimationAgentType]
        List of agent types, with simulation results but no solution.
    '''
    if use_cohorts:
        type_list = makeMultiTypeWithCohorts(params)
    else:
        type_list = makeMultiTypeSimple(params)
        
    multiThreadCommands(type_list,['estimationAction()'],num_jobs=5)
    return type_list


def calcSimulatedMoments(type_list,return_as_list):
    '''
    Calculate simulated counterparts to all of the data moments.
    
    Parameters
    ----------
    type_list : [EstimationAgentType]
        List of agent types, with simulation results but no solution.
    return_as_list : bool
        Indicator for whether the moments should be returned as a list of arrays
        or already aggregated into a single vector.
        
    Returns
    -------
    all_moments : np.array or [np.array]
        Very long 1D array with all simulated moments OR list of arrays.
    '''
    # Combine simulated data across all types
    aLvlHist = np.concatenate([this_type.aLvlNow_hist for this_type in type_list],axis=1)
    hLvlHist = np.concatenate([this_type.hLvlNow_hist for this_type in type_list],axis=1)
    OOPhist  = np.concatenate([this_type.OOPmedNow_hist for this_type in type_list],axis=1)
    MortHist = np.concatenate([this_type.DiePrbNow_hist for this_type in type_list],axis=1)
    WeightHist = np.concatenate([this_type.CumLivPrb_hist for this_type in type_list],axis=1)
    
    # Combine data labels across types
    HealthTert = np.concatenate([this_type.HealthTert for this_type in type_list])
    HealthQuint = np.concatenate([this_type.HealthQuint for this_type in type_list])
    WealthQuint = np.concatenate([this_type.WealthQuint for this_type in type_list])
    IncQuint = np.concatenate([this_type.IncQuintLong for this_type in type_list])
    Sex = np.concatenate([this_type.SexLong for this_type in type_list])
    
    # Combine in-data-span masking array across all types
    InDataSpan = np.concatenate([this_type.InDataSpanArray for this_type in type_list],axis=1)
    
    # Determine eligibility to be used for various purposes
    Active = hLvlHist > 0.
    
    # Calculate the change in health from period to period for all simulated agents
    DeltaHealth = np.zeros_like(hLvlHist)
    hNext = np.zeros_like(hLvlHist)
    hNext[:-1,:] = hLvlHist[1:,:]
    DeltaHealth[:-1,:] = hNext[:-1,:] - hLvlHist[:-1,:]
    
    # Initialize arrays to hold simulated moments
    OOPbyAge = np.zeros(15)
    StDevOOPbyAge = np.zeros(15)
    MortByAge = np.zeros(15)
    StDevDeltaHealthByAge = np.zeros(15)
    StDevOOPbyHealthAge = np.zeros((3,15))
    StDevDeltaHealthByHealthAge = np.zeros((3,15))
    HealthBySexHealthAge = np.zeros((2,3,15))
    OOPbySexHealthAge = np.zeros((2,3,15))
    MortBySexHealthAge = np.zeros((2,3,15))
    WealthByIncAge = np.zeros((5,15))
    HealthByIncAge = np.zeros((5,15))
    OOPbyIncAge = np.zeros((5,15))
    WealthByIncWealthAge = np.zeros((5,5,15))
    HealthByIncWealthAge = np.zeros((5,5,15))
    OOPbyIncWealthAge = np.zeros((5,5,15))
    MortByHealthAge = np.zeros((5,15))
    HealthByHealthAge = np.zeros((5,15))
    WealthByHealthAge = np.zeros((5,15))
    
    # Make large 1D vectors for the health transition and OOP regressions
    THESE = np.logical_and(Active,InDataSpan)
    T = aLvlHist.shape[0]
    N = aLvlHist.shape[1]
    hNext_reg = hNext[THESE]
    OOP_reg = OOPhist[THESE]
    h_reg = hLvlHist[THESE]
    hSq_reg = h_reg**2
    AgeHist = np.tile(np.reshape(np.arange(T),(T,1)),(1,N))
    a_reg = AgeHist[THESE]
    aSq_reg = a_reg**2
    SexHist = np.tile(np.reshape(Sex.astype(int),(1,N)),(T,1))
    s_reg = SexHist[THESE]
    const_reg = np.ones_like(s_reg)
    del AgeHist, SexHist
    IQhist = np.tile(np.reshape(IncQuint,(1,N)),(T,1))
    IQ_reg = IQhist[THESE]
    WQhist = np.tile(np.reshape(WealthQuint,(1,N)),(T,1))
    WQ_reg = WQhist[THESE]
    del IQhist, WQhist
    WeightNext = np.zeros_like(WeightHist)
    WeightNext[:-1,:] = WeightHist[1:,:]
    weight_reg = WeightNext[THESE]
    del WeightNext
    
    # Make arrayw of inc-wealth quintile indicators
    IQbool_reg = np.zeros((5,OOP_reg.size))
    for i in range(5):
        IQbool_reg[i,:] = IQ_reg == i+1
    WQbool_reg = np.zeros((5,OOP_reg.size))
    for j in range(5):
        WQbool_reg[j,:] = WQ_reg == j+1
    IWQbool_reg = np.zeros((25,h_reg.size))
    k = 0
    for i in range(5):
        for j in range(5):
            these = np.logical_and(IQbool_reg[i,:],WQbool_reg[j,:])
            IWQbool_reg[k,these] = 1.0
            k += 1
    
    # Regress health next on sex, health, age, and quintile dummies
    regressors = np.transpose(np.vstack([const_reg,s_reg,h_reg,hSq_reg,a_reg,aSq_reg,IWQbool_reg[3:,:]]))
    health_model = WLS(hNext_reg,regressors,weights=weight_reg)
    health_results = health_model.fit()
    AvgHealthResidualByIncWealth = np.reshape(np.concatenate([[0.,0.,0.],health_results.params[-22:]]),(5,5))
    
    # Regress OOP on sex, health, age, and quintile dummies
    #regressors = np.transpose(np.vstack([const_reg,s_reg,h_reg,hSq_reg,a_reg,aSq_reg,IQbool_reg[1:,:],WQbool_reg[1:,:]]))
    OOP_model = WLS(OOP_reg,regressors,weights=weight_reg)
    OOP_results = OOP_model.fit()
    #AvgOOPResidualByIncWealth = np.reshape(np.concatenate([[0.],OOP_results.params[6:10],[0.],OOP_results.params[10:]]),(2,5))
    AvgOOPResidualByIncWealth = np.reshape(np.concatenate([[0.,0.,0.],OOP_results.params[-22:]]),(5,5))
    
    # Loop through ages, sexes, quintiles, and health to fill in simulated moments
    for t in range(15):
        # Calculate mean and stdev of OOP medical spending, mortality rate, and stdev delta health by age
        THESE = np.logical_and(Active[t,:],InDataSpan[t,:])
        OOP = OOPhist[t,THESE]
        Weight = WeightHist[t+1,THESE]
        WeightSum = np.sum(Weight)
        MeanOOP = np.dot(OOP,Weight)/WeightSum
        OOPbyAge[t] = MeanOOP
        OOPsqDevFromMean = (OOP - MeanOOP)**2
        StDevOOPbyAge[t] = np.sqrt(np.dot(OOPsqDevFromMean,Weight)/WeightSum)
        Mort = MortHist[t+1,THESE]
        MortByAge[t] = np.dot(Mort,Weight)/WeightSum
        HealthChange = DeltaHealth[t,THESE]
        MeanHealthChange = np.dot(HealthChange,Weight)/WeightSum
        HealthChangeSqDevFromMean = (HealthChange - MeanHealthChange)**2
        StDevDeltaHealthByAge[t] = np.sqrt(np.dot(HealthChangeSqDevFromMean,Weight)/WeightSum)
        
        for h in range(5):
            right_health = HealthQuint == (h+1)
            these = np.logical_and(THESE,right_health)
            Mort = MortHist[t+1,these]
            Health = hLvlHist[t+1,these]
            Wealth = aLvlHist[t,these]
            Weight = WeightHist[t+1,these]
            WeightSum = np.sum(Weight)
            MortByHealthAge[h,t] = np.dot(Mort,Weight)/WeightSum
            HealthByHealthAge[h,t] = np.dot(Health,Weight)/WeightSum
            WealthByHealthAge[h,t] = getPercentiles(Wealth,weights=Weight)
        
        for h in range(3):
            # Calculate stdev OOP medical spending and stdev delta health by health by age
            right_health = HealthTert==(h+1)
            these = np.logical_and(THESE,right_health)
            OOP = OOPhist[t,these]
            Weight = WeightHist[t+1,these]
            WeightSum = np.sum(Weight)
            MeanOOP = np.dot(OOP,Weight)/WeightSum
            OOPbySexHealthAge[0,h,t] = MeanOOP
            OOPsqDevFromMean = (OOP - MeanOOP)**2
            StDevOOPbyHealthAge[h,t] = np.sqrt(np.dot(OOPsqDevFromMean,Weight)/WeightSum)
            HealthChange = DeltaHealth[t,these]
            MeanHealthChange = np.dot(HealthChange,Weight)/WeightSum
            HealthChangeSqDevFromMean = (HealthChange - MeanHealthChange)**2
            StDevDeltaHealthByHealthAge[h,t] = np.sqrt(np.dot(HealthChangeSqDevFromMean,Weight)/WeightSum)
            
            for s in range(2):
                # Calculate mean OOP medical spending and mortality by sex by health by age
                right_sex = Sex==s
                those = np.logical_and(these,right_sex)
                OOP = OOPhist[t,those]
                Health = hLvlHist[t+1,those]
                Mort = MortHist[t+1,those]
                Weight = WeightHist[t+1,those]
                WeightSum = np.sum(Weight)
                #MeanOOP = np.dot(OOP,Weight)/WeightSum
                #OOPbySexHealthAge[s,h,t] = MeanOOP
                HealthBySexHealthAge[s,h,t] = np.dot(Health,Weight)/WeightSum
                MortBySexHealthAge[s,h,t] = np.dot(Mort,Weight)/WeightSum
        
        for s in range(2):
            # Calculate mean OOP medical spending by sex by age
            right_sex = Sex==s
            these = np.logical_and(THESE,right_sex)
            OOP = OOPhist[t,these]
            Weight = WeightHist[t+1,these]
            WeightSum = np.sum(Weight)
            MeanOOP = np.dot(OOP,Weight)/WeightSum
            OOPbySexHealthAge[1,s,t] = MeanOOP
                
        for i in range(5):
            # Calculate median wealth, mean health, and mean OOP medical spending by income quintile by age
            right_inc = IncQuint == i+1
            these = np.logical_and(THESE,right_inc)
            OOP = OOPhist[t,these]
            Wealth = aLvlHist[t,these]
            Health = hLvlHist[t+1,these]
            Weight = WeightHist[t+1,these]
            WeightSum = np.sum(Weight)
            WealthByIncAge[i,t] = getPercentiles(Wealth,weights=Weight)
            HealthByIncAge[i,t] = np.dot(Health,Weight)/WeightSum
            OOPbyIncAge[i,t] = np.dot(OOP,Weight)/WeightSum
            
            for j in range(5):
                # Calculate median wealth, mean health, and mean OOP medical spending by income quintile by wealth quintile by age
                right_wealth = WealthQuint == j+1
                those = np.logical_and(these,right_wealth)
                OOP = OOPhist[t,those]
                Wealth = aLvlHist[t,those]
                Health = hLvlHist[t+1,those]
                Weight = WeightHist[t+1,those]
                WeightSum = np.sum(Weight)
                WealthByIncWealthAge[i,j,t] = getPercentiles(Wealth,weights=Weight)
                HealthByIncWealthAge[i,j,t] = np.dot(Health,Weight)/WeightSum
                OOPbyIncWealthAge[i,j,t] = np.dot(OOP,Weight)/WeightSum
                
    # Aggregate moments into a single vector and return (or return moments separately in a list)
    if return_as_list:
       all_moments = [
                OOPbyAge,
                StDevOOPbyAge,
                MortByAge,
                StDevDeltaHealthByAge,
                StDevOOPbyHealthAge,
                StDevDeltaHealthByHealthAge,
                HealthBySexHealthAge,
                OOPbySexHealthAge,
                MortBySexHealthAge,
                WealthByIncAge,
                HealthByIncAge,
                OOPbyIncAge,
                WealthByIncWealthAge,
                HealthByIncWealthAge,
                OOPbyIncWealthAge,
                AvgHealthResidualByIncWealth,
                AvgOOPResidualByIncWealth,
                MortByHealthAge,
                HealthByHealthAge,
                WealthByHealthAge
                ]
    else: 
        all_moments = np.concatenate([
                OOPbyAge,
                StDevOOPbyAge,
                MortByAge,
                StDevDeltaHealthByAge,
                StDevOOPbyHealthAge.flatten(),
                StDevDeltaHealthByHealthAge.flatten(),
                HealthBySexHealthAge.flatten(),
                OOPbySexHealthAge.flatten(),
                MortBySexHealthAge.flatten(),
                WealthByIncAge.flatten(),
                HealthByIncAge.flatten(),
                OOPbyIncAge.flatten(),
                WealthByIncWealthAge.flatten(),
                HealthByIncWealthAge.flatten(),
                OOPbyIncWealthAge.flatten(),
                AvgHealthResidualByIncWealth.flatten(),
                AvgOOPResidualByIncWealth.flatten(),
                MortByHealthAge.flatten(),
                HealthByHealthAge.flatten(),
                WealthByHealthAge.flatten()
                ])
    return all_moments


def pseudoEstHealthProdParams(type_list,return_as_list):
    '''
    Run a pseudo-estimation of the health production parameters given simulation
    results from a candidate parameter set.  Similar to the pre-estimation,
    this procedure tries to fit the "health residual" and "OOP residual" moments,
    but substitutes "health produced" for the former.
    
    Parameters
    ----------
    type_list : [EstimationAgentType]
        List of agent types, with simulation results but no solution.
    return_as_list : bool
        Indicator for whether the moments should be returned as a list of arrays
        or already aggregated into a single vector.
        
    Returns
    -------
    TBD
    '''
    # Combine simulated data across all types
    hLvlHist = np.concatenate([this_type.hLvlNow_hist for this_type in type_list],axis=1)
    MedLvlHist  = np.concatenate([this_type.MedLvlNow_hist for this_type in type_list],axis=1)
    WeightHist = np.concatenate([this_type.CumLivPrb_hist for this_type in type_list],axis=1)
    RatioHist = np.concatenate([this_type.RatioNow_hist for this_type in type_list],axis=1)
    CopayMedHist = np.concatenate([this_type.CopayMedNow_hist for this_type in type_list],axis=1)
    CopayInvstHist = np.concatenate([this_type.CopayInvstNow_hist for this_type in type_list],axis=1)
    
    # Combine data labels across types
    WealthQuint = np.concatenate([this_type.WealthQuint for this_type in type_list])
    IncQuint = np.concatenate([this_type.IncQuintLong for this_type in type_list])
    Sex = np.concatenate([this_type.SexLong for this_type in type_list])
    
    # Combine in-data-span masking array across all types
    InDataSpan = np.concatenate([this_type.InDataSpanArray for this_type in type_list],axis=1)
    
    # Determine eligibility to be used for various purposes
    Active = hLvlHist > 0.
    
    # Calculate the change in health from period to period for all simulated agents
    DeltaHealth = np.zeros_like(hLvlHist)
    hNext = np.zeros_like(hLvlHist)
    hNext[:-1,:] = hLvlHist[1:,:]
    DeltaHealth[:-1,:] = hNext[:-1,:] - hLvlHist[:-1,:]
    
    # Make large 1D vectors for the health transition and OOP regressions
    THESE = np.logical_and(Active,InDataSpan)
    T = hLvlHist.shape[0]
    N = hLvlHist.shape[1]
    Med_reg = MedLvlHist[THESE]
    Ratio_reg = RatioHist[THESE]
    CopayMed_reg = CopayMedHist[THESE]
    CopayInvst_reg = CopayInvstHist[THESE]
    h_reg = hLvlHist[THESE]
    hSq_reg = h_reg**2
    AgeHist = np.tile(np.reshape(np.arange(T),(T,1)),(1,N))
    a_reg = AgeHist[THESE]
    aSq_reg = a_reg**2
    SexHist = np.tile(np.reshape(Sex.astype(int),(1,N)),(T,1))
    s_reg = SexHist[THESE]
    const_reg = np.ones_like(s_reg)
    del AgeHist, SexHist
    IQhist = np.tile(np.reshape(IncQuint,(1,N)),(T,1))
    IQ_reg = IQhist[THESE]
    WQhist = np.tile(np.reshape(WealthQuint,(1,N)),(T,1))
    WQ_reg = WQhist[THESE]
    del IQhist, WQhist
    WeightNext = np.zeros_like(WeightHist)
    WeightNext[:-1,:] = WeightHist[1:,:]
    weight_reg = WeightNext[THESE]
    del WeightNext
    
    # Make arrays of inc-wealth quintile indicators
    IQbool_reg = np.zeros((5,h_reg.size))
    for i in range(5):
        IQbool_reg[i,:] = IQ_reg == i+1
    WQbool_reg = np.zeros((5,h_reg.size))
    for j in range(5):
        WQbool_reg[j,:] = WQ_reg == j+1
    IWQbool_reg = np.zeros((25,h_reg.size),dtype=bool)
    k = 0
    for i in range(5):
        for j in range(5):
            these = np.logical_and(IQbool_reg[i,:],WQbool_reg[j,:])
            IWQbool_reg[k,these] = 1.0
            k += 1
            
    # Define regressors for the OOP model
    #regressors = np.transpose(np.vstack([const_reg,s_reg,h_reg,hSq_reg,a_reg,aSq_reg,IQbool_reg[1:,:],WQbool_reg[1:,:]]))
    regressors = np.transpose(np.vstack([const_reg,s_reg,h_reg,hSq_reg,a_reg,aSq_reg,IWQbool_reg[3:,:]]))
    
    # Define data moments and data weights
    TempDataMoments = DataMoments[-50:]
    TempWeights = MomentWeights[-50:,-50:]
    TempMask = MomentMask[-50:]
            
    def makePseudoEstMoments(p0,p1,p2):
        '''
        Calculates simulated moments for the health production parameter pseudo estimation.
        '''
        # Define health production functions
        tempw = 2. - np.exp(p0)
        HealthProd0 = (tempw-1.)/(tempw-2.)
        tempx = np.exp(p1) # Slope of health production function at iLvl=0
        tempy = -np.exp(p1+p2) # Curvature of health prod at iLvl=0
        HealthProd2 = tempx/tempy*(HealthProd0-1.)
        HealthProdFunc = lambda i : tempx/HealthProd0*((i*HealthProd2**((1.-HealthProd0)/HealthProd0) + HealthProd2**(1./HealthProd0))**HealthProd0 - HealthProd2)
        MargHealthProdInvFunc = lambda q : HealthProd2*((q/tempx)**(1./(HealthProd0-1.)) -1.)
        
        # Calculate health investment, health produced, and OOP medical spending
        HealthInv = np.maximum(MargHealthProdInvFunc(Ratio_reg),0.0)
        HealthInv[Ratio_reg == 0.] = 0.
        HealthProd = HealthProdFunc(HealthInv)
        OOP_reg = HealthInv*CopayInvst_reg + Med_reg*CopayMed_reg
        
        # Regress OOP medical spending on sex, age, health, quintile dummies
        OOP_model = WLS(OOP_reg,regressors,weights=weight_reg)
        OOP_results = OOP_model.fit()
        #AvgOOPResidualByIncWealth = np.concatenate([[0.],OOP_results.params[6:10],[0.],OOP_results.params[10:]])
        AvgOOPResidualByIncWealth = np.concatenate([[0.,0.,0.],OOP_results.params[-22:]])
        
        # Calculate average health produced by income-wealth quintile
        AvgHealthResidualByIncWealth = np.zeros(25)
        for i in range(25):
            these = IWQbool_reg[i,:]
            AvgHealthResidualByIncWealth[i] = np.dot(HealthProd[these],weight_reg[these])/np.sum(weight_reg[these])
            
        # Assemble simulated moments
        SimMoments = np.concatenate([AvgHealthResidualByIncWealth,AvgOOPResidualByIncWealth])
        return SimMoments
        
    def pseudoEstObjFunc(p0,p1,p2):
        '''
        Objective function for the health production parameter pseudo estimation.
        Takes in the three health production parameters, returns weighted distance
        for *only* the health residual and OOP residual moments.  Health residuals
        are calculated as health produced.
        '''
        SimMoments = makePseudoEstMoments(p0,p1,p2)
        MomentDifferences = np.reshape((SimMoments - TempDataMoments)*TempMask,(50,1))
        weighted_moment_sum = np.dot(np.dot(MomentDifferences.transpose(),TempWeights),MomentDifferences)[0,0]
        print(weighted_moment_sum)
        return weighted_moment_sum
    
    # Run the health production parameter pseudo-estimation
    temp_f = lambda x : pseudoEstObjFunc(x[0],x[1],x[2])
    guess = np.array([-16.0,-2.0,1.5])
    opt_params = minimizeNelderMead(temp_f,guess)
    print(opt_params)
    opt_moments = makePseudoEstMoments(opt_params[0],opt_params[1],opt_params[2])
    if return_as_list:
        opt_moments = [np.reshape(opt_moments[0:25],(5,5)),np.reshape(opt_moments[25:],(5,5))]
    return opt_moments


def calcStdErrs(params,use_cohorts,which,eps):
    '''
    Calculate standard errors of the estimated parameters for the Ounce of Prevention
    project.  Approximates the Hessian of the objective function as the inner product
    of the Jacobian of the moment difference function (with the weighting matrix).
    
    Parameters
    ----------
    params : dict
        The dictionary to be used to construct each of the types.
    use_cohorts : bool
        Indicator for whether to separately solve and simulate the 15 cohorts.
    which : np.array
        Length 33 boolean array indicating which parameters should get std errs.
    eps : float
        Relative perturbation to each parameter to calculate numeric derivatives.
        
    Returns
    -------
    StdErrVec : np.array
        Vector of length np.sum(which) with standard errors for the indicated structural parameters.
    ParamCovarMatrix : np.arrau
        Square array of covariances for the indicated structural parameters.
    '''
    # Initialize an array of numeric derivatives of moment differences
    N = np.sum(which)
    MomentDerivativeArray = np.zeros((N,MomentCount))
    
    # Make a dictionary of parameters
    base_param_dict = convertVecToDict(params)
    
    # Calculate the vector of moment differences for the estimated parameters
    TypeList = processSimulatedTypes(base_param_dict,use_cohorts)
    SimulatedMoments = calcSimulatedMoments(TypeList,False)
    BaseMomentDifferences = (SimulatedMoments - DataMoments)
    print('Found moment differences for base parameter vector')
    
    # Loop through the parameters, perturbing each one and calculating moment derivatives
    n = 0
    for i in range(33):
        if which[i]:
            params_now = copy(params)
            this_eps = params[i]*eps
            this_param = params[i] + this_eps
            params_now[i] = this_param
            this_param_dict = convertVecToDict(params_now)
            TypeList = processSimulatedTypes(this_param_dict,use_cohorts)
            SimulatedMoments = calcSimulatedMoments(TypeList,False)
            MomentDifferences = (SimulatedMoments - DataMoments)*MomentMask
            MomentDerivativeArray[n,:] = (MomentDifferences - BaseMomentDifferences)/this_eps
            n += 1
            print('Finished perturbing parameter ' + str(n) + ' of ' + str(N) + ', NaN count = ' + str(np.sum(np.isnan(MomentDerivativeArray[n-1,:]))))
            
    # Calculate standard errors by finding the variance-covariance matrix for the parameters
    scale_fac = 1. + 1./Params.basic_estimation_dict['DataToSimRepFactor']
    ParamCovMatrix = np.linalg.inv(np.dot(MomentDerivativeArray,np.dot(MomentWeights,MomentDerivativeArray.transpose())))
    ParamCovMatrix *= scale_fac
    StdErrVec = np.sqrt(np.diag(ParamCovMatrix))
    return StdErrVec, ParamCovMatrix, MomentDerivativeArray


def findBracketingInterval(f,x0,init_eps=0.1):
    '''
    Finds a bracketing interval that contains a zero of continuous function f: R-->R
    with x0 as one end point.  This bracketing interval can be used in a rootfinding
    routine that requires this input.
    
    Parameters
    ----------
    f : function
        Function to find a zero bracketing interval; assumed continuous.
    x0 : float
        One end of the bracketing interval.
    init_eps : float
        Initial perturbation to x0 when searching for the other bracket end.
        
    Returns
    -------
    [x0,x1] or [x1,x0] : [float,float]
        Bracketing interval for a zero of f, with lower element in first position.
    '''
    eps = init_eps
    found_it = False
    f0 = f(x0)
    f0_positive = f0 > 0.
    
    while not found_it:
        x1 = x0 + eps
        f1 = f(x1)
        f1_positive = f1 > 0.
        found_it = np.logical_xor(f0_positive,f1_positive)
        if not found_it:
            eps *= np.e
            
    if x1 > x0:
        return [x0,x1]
    else:
        return [x1,x0]
    
    
def calcStdErrsAlt(params,use_cohorts,which,f_dev):
    '''
    Calculate standard errors of the estimated parameters for the Ounce of Prevention
    project.  Approximates the Hessian of the objective function by searching for the
    quadratic coefficients that best fit the objective function near the estimated values.
    
    Parameters
    ----------
    params : dict
        The dictionary to be used to construct each of the types.
    use_cohorts : bool
        Indicator for whether to separately solve and simulate the 15 cohorts.
    which : np.array
        Length 33 boolean array indicating which parameters should get std errs.
    f_dev : float
        Positive increase in the objective function to search for each parameter.
        
    Returns
    -------
    StdErrVec : np.array
        Vector of length np.sum(which) with standard errors for the indicated structural parameters.
    ParamCovarMatrix : np.arrau
        Square array of covariances for the indicated structural parameters.
    '''
    N = np.sum(which)
    fOpt = objectiveFunctionWrapper(params)
    x_dev_vec = np.zeros(N)
    coeff_matrix = np.zeros((N,N))
    
    n = 0
    for i in range(33):
        if which[i]:
            def temp_f(x):
                eps = np.zeros(33)
                eps[i] = x
                return objectiveFunctionWrapper(params+eps) - fOpt - f_dev
            
            bracket = findBracketingInterval(temp_f,0.0,0.1*params[i])
            x_dev_vec[n] = brentq(temp_f, bracket[0], bracket[1], xtol=2e-12, rtol=1e-6, maxiter=100)
            coeff_matrix[n,n] = f_dev/x_dev_vec[n]**2
            print('Found diagonal coefficient for parameter ' + str(n+1) + ' of ' + str(N) + '.')
            
            nn = 0
            for j in range(i):
                if which[j]:
                    eps1 = np.zeros(33)
                    eps1[i] = x_dev_vec[n]
                    eps2 = np.zeros(33)
                    eps2[j] = x_dev_vec[nn]
                    temp_params = params + eps1 + eps2
                    f2 = objectiveFunctionWrapper(temp_params) - fOpt - 2*f_dev
                    diag_coeff = f2/(2*x_dev_vec[n]*x_dev_vec[nn])
                    coeff_matrix[n,nn] = diag_coeff
                    coeff_matrix[nn,n] = diag_coeff
                    nn += 1
            print('Found ' + str(2*n) + ' off-diagonal coefficients for parameter ' + str(n+1) + ' of ' + str(N) + '.')
                    
            n += 1
    
    hessian = 2*coeff_matrix
    ParamCovarMatrix = np.linalg.inv(hessian)
    StdErrVec = np.sqrt(np.diag(ParamCovarMatrix))
    
    return StdErrVec, ParamCovarMatrix
    

def objectiveFunction(params,use_cohorts,return_as_list):
    '''
    The objective function for the ounce of prevention estimation.  Takes a dictionary
    of parameters and a boolean indicator for whether to break sample up by cohorts.
    Returns a single real representing the weighted sum of squared moment distances.
    
    Parameters
    ----------
    params : dict
        The dictionary to be used to construct each of the types.
    use_cohorts : bool
        Indicator for whether to separately solve and simulate the 15 cohorts.
    return_as_list : bool
        Indicator for whether the moments should be returned as a list of arrays
        or already aggregated into a single vector.
        
    Returns
    -------
    weighted_moment_sum : float
        Weighted sum of squared moment differences between data and simulation.
        OR
    SimulatedMoments : [np.array]
        List of all moments, separated by types into different arrays.
    '''
    TypeList = processSimulatedTypes(params,use_cohorts)
    SimulatedMoments = calcSimulatedMoments(TypeList,return_as_list)
    
    if False: # Set this block to true to run the "pseudo estimation" for health production parameters
        HealthProdMoments = pseudoEstHealthProdParams(TypeList,return_as_list)
        if return_as_list:
            SimulatedMoments[-2] = HealthProdMoments[0]
            SimulatedMoments[-1] = HealthProdMoments[1]
        else:
            SimulatedMoments[-50:] = HealthProdMoments
    
    if return_as_list:
        return SimulatedMoments
    else:
        MomentDifferences = np.reshape((SimulatedMoments - DataMoments)*MomentMask,(MomentCount,1))
        weighted_moment_sum = np.dot(np.dot(MomentDifferences.transpose(),MomentWeights),MomentDifferences)[0,0]
        print(weighted_moment_sum)
        return weighted_moment_sum


def writeParamsToFile(param_vec,filename):
    '''
    Store the current parameter files in a txt file so they can be recovered in
    case of a computer crash (etc).
    
    Parameters
    ----------
    param_vec : np.array
        Size 33 array of structural parameters.
    filename : str
        Name of file in which to write the current parameters.
        
    Returns
    -------
    None
    '''
    write_str = 'current_param_vec = np.array([ \n'
    for i in range(param_vec.size):
        write_str += '    ' + str(param_vec[i]) + ',  # ' + str(i) + ' ' + Params.param_names[i] + '\n'
    write_str += ']) \n'
    with open('./Data/' + filename,'wb') as f:
        f.write(write_str)
        f.close()
        
        
def convertVecToDict(param_vec):
    '''
    Converts a 33 length vector of parameters to a dictionary that can be used
    by the estimator or standard error calculator.
    '''
    struct_params = {
        'CRRA' : param_vec[0],
        'DiscFac' : param_vec[1],
        'MedCurve' : param_vec[2],
        'LifeUtility' : param_vec[3],
        'MargUtilityShift' : param_vec[4],
        'Cfloor' : param_vec[5],
        'Bequest0' : param_vec[6],
        'Bequest1' : param_vec[7],
        'MedShkMean0' : param_vec[8],
        'MedShkMeanSex' : param_vec[9],
        'MedShkMeanAge' : param_vec[10],
        'MedShkMeanAgeSq' : param_vec[11],
        'MedShkMeanHealth' : param_vec[12],
        'MedShkMeanHealthSq' : param_vec[13],
        'MedShkStd0' : param_vec[14],
        'MedShkStd1' : param_vec[15],
        'HealthNext0' : param_vec[16],
        'HealthNextSex' : param_vec[17],
        'HealthNextAge' : param_vec[18],
        'HealthNextAgeSq' : param_vec[19],
        'HealthNextHealth' : param_vec[20],
        'HealthNextHealthSq' : param_vec[21],
        'HealthShkStd0' : param_vec[22],
        'HealthShkStd1' : param_vec[23],
        'LogJerk' : param_vec[24],
        'LogSlope' : param_vec[25],
        'LogCurve' : param_vec[26],
        'Mortality0' : param_vec[27],
        'MortalitySex' : param_vec[28],
        'MortalityAge' : param_vec[29],
        'MortalityAgeSq' : param_vec[30],
        'MortalityHealth' : param_vec[31],
        'MortalityHealthSq' : param_vec[32]
    }
    param_dict = copy(Params.basic_estimation_dict)
    param_dict.update(struct_params)
    return param_dict


def objectiveFunctionWrapper(param_vec):
    '''
    Wrapper funtion around the objective function so that it can be used with
    optimization routines.  Takes a single 1D array as input, returns a single float.
    
    Parameters
    ----------
    param_vec : np.array
        1D array of structural parameters to be estimated.  Should have size 33.
        
    Returns
    -------
    weighted_moment_sum : float
        Weighted sum of squared moment differences between data and simulation.
    '''
    # Make a dictionary with structural parameters for testing
    these_params = convertVecToDict(param_vec)
    
    # Run the objective function with the newly created dictionary
    use_cohorts = Data.use_cohorts
    weighted_moment_sum = objectiveFunction(these_params,use_cohorts,plot_model_fit)
    
    # Write the current parameters to a file if a certain number of function calls have happened
    Params.func_call_count += 1
    if np.mod(Params.func_call_count,Params.store_freq) == 0:
        writeParamsToFile(param_vec,'ParameterStatus.txt')
    
    return weighted_moment_sum



if __name__ == '__main__':

#    i=0
#    param_dict = convertVecToDict(Params.test_param_vec)
#    MyTypes = makeMultiTypeSimple(param_dict)
#    t_start = clock()
#    MyTypes[i].estimationAction()
#    t_end = clock()
#    print('Processing one agent type took ' + str(t_end-t_start) + ' seconds.')
#
#    t=0
#    bMax = 100.
#    MyTypes[i].plotxFuncByHealth(t,Dev=0.,bMax=bMax)
#    MyTypes[i].plotxFuncByMedShk(t,hLvl=0.9,bMax=bMax)
#    MyTypes[i].plotcFuncByHealth(t,Dev=0.,bMax=bMax)
#    MyTypes[i].plotcFuncByMedShk(t,hLvl=0.9,bMax=bMax)
#    MyTypes[i].plotMedFuncByHealth(t,Dev=0.,bMax=bMax)
#    MyTypes[i].plotMedFuncByMedShk(t,hLvl=0.9,bMax=bMax)
#    MyTypes[i].plotiFuncByHealth(t,Dev=0.,bMax=bMax)
#    MyTypes[i].plotiFuncByMedShk(t,hLvl=0.9,bMax=bMax)
#    MyTypes[i].plotvFuncByHealth(t,bMax=bMax)
#    MyTypes[i].plotdvdbFuncByHealth(t,bMax=bMax)
#    #MyTypes[i].plotdvdhFuncByHealth(t,bMax=bMax)
#
#    MyTypes[i].plot2DfuncByHealth('TotalMedPDVfunc',t,bMax=bMax)
#    MyTypes[i].plot2DfuncByWealth('TotalMedPDVfunc',t)
#    MyTypes[i].plot2DfuncByHealth('OOPmedPDVfunc',t,bMax=bMax)
#    MyTypes[i].plot2DfuncByWealth('OOPmedPDVfunc',t)
#    MyTypes[i].plot2DfuncByHealth('SubsidyPDVfunc',t,bMax=bMax)
#    MyTypes[i].plot2DfuncByWealth('SubsidyPDVfunc',t)
#    MyTypes[i].plot2DfuncByHealth('MedicarePDVfunc',t,bMax=bMax)
#    MyTypes[i].plot2DfuncByWealth('MedicarePDVfunc',t)
#    MyTypes[i].plot2DfuncByHealth('WelfarePDVfunc',t,bMax=bMax)
#    MyTypes[i].plot2DfuncByWealth('WelfarePDVfunc',t)
#    MyTypes[i].plot2DfuncByHealth('GovtPDVfunc',t,bMax=bMax)
#    MyTypes[i].plot2DfuncByWealth('GovtPDVfunc',t)
#    MyTypes[i].plot2DfuncByHealth('ExpectedLifeFunc',t,bMax=bMax)
#    MyTypes[i].plot2DfuncByWealth('ExpectedLifeFunc',t)
    
    
#    t_start = clock()
#    MyTypes = processSimulatedTypes(Params.test_params,False)
#    t_end = clock()
#    print('Processing ten agent types took ' + str(t_end-t_start) + ' seconds.')
#    
#    t_start = clock()
#    X = calcSimulatedMoments(MyTypes)
#    t_end = clock()
#    print('Calculating moments took ' + str(t_end-t_start) + ' seconds.')
    
#    t=0
#    bMax=200.
#    
#    for j in range(10):
#        MyTypes[j].plotxFuncByHealth(t,MedShk=1.0,bMax=bMax)


    # Choose what kind of work to do:
    test_obj_func = True
    plot_model_fit = True
    save_figs = True
    perturb_one_param = False
    perturb_two_params = False
    estimate_model = False
    calc_std_errs = False
    calc_std_errs_alt = False
    

    if test_obj_func:
        t_start = clock()
        X = objectiveFunctionWrapper(Params.test_param_vec)
        t_end = clock()
        print('One objective function evaluation took ' + str(t_end-t_start) + ' seconds.')
    
    if plot_model_fit:
        AgeVec = np.linspace(67.,95.,num=15)
        health_colors = ['b','r','g']
        sex_colors = ['r','b']
        income_colors = ['b','r','g','m','c']
        
        # Plot model fit of mean out of pocket medical spending by age
        plt.plot(AgeVec,X[0],'-m')
        plt.plot(AgeVec,X[1],'-c')
        plt.plot(AgeVec,Data.OOPbyAge,'.m')
        plt.plot(AgeVec,Data.StDevOOPbyAge,'.c')
        plt.title('Mean and stdev out-of-pocket medical spending [3(a), 4(a)]')
        plt.ylabel(r'OOP expenses $o_{it}$, \$10,000 USD (y2000)')
        plt.xlabel('Age')
        plt.ylim([0.,1.75])
        plt.legend(['Mean','Standard deviation'],loc=2)
        if save_figs:
            plt.savefig('./Figures/OOPbyAge.pdf')
        plt.show()
        
        # Plot model fit of mean out of pocket medical spending by age-health for all
        for h in range(3):
            plt.plot(AgeVec,X[7][0,h,:], health_colors[h] + '-')
        for h in range(3):
            plt.plot(AgeVec,Data.OOPbySexHealthAge[0,h,:], '.' + health_colors[h])
        plt.title('Mean out-of-pocket medical spending by health [3(b)]')
        plt.ylabel(r'OOP expenses $o_{it}$, \$10,000 USD (y2000)')
        plt.xlabel('Age')
        plt.legend(['Bottom tertile','Middle tertile','Top tertile'],loc=2)
        if save_figs:
            plt.savefig('./Figures/OOPbyHealthAge.pdf')
        plt.show()
        
        # Plot model fit of mean out of pocket medical spending by age-sex
        for s in range(2):
            plt.plot(AgeVec,X[7][1,s,:], sex_colors[s] + '-')
        for s in range(2):
            plt.plot(AgeVec,Data.OOPbySexHealthAge[1,s,:], '.' + sex_colors[s])
        plt.title('Mean out-of-pocket medical spending by sex [3(c)]')
        plt.ylabel(r'OOP expenses $o_{it}$, \$10,000 USD (y2000)')
        plt.xlabel('Age')
        plt.legend(['Women','Men'],loc=2)
        if save_figs:
            plt.savefig('./Figures/OOPbySexAge.pdf')
        plt.show()
        
        # Plot model fit of "OOP coefficient" by wealth and income
        for i in range(5):
            plt.plot(np.arange(1,6),X[16][i,:], income_colors[i] + '-')
        for i in range(5):
            plt.plot(np.arange(1,6),Data.AvgOOPResidualByIncWealth[i,:], '.' + income_colors[i])
        plt.xlabel('Wealth quintile')
        plt.ylabel(r'Coeff on income-wealth quintile $\daleth_d[k,\ell]$, \$10k')
        plt.xticks(np.arange(1,6),['Bottom','Second','Third','Fourth','Top'])
        plt.title('Out-of-pocket medical spending by income and wealth quintile, [7(b)]')
        if save_figs:
            plt.savefig('./Figures/OOPcoeffByIncWealth.pdf')
        plt.show()
        
        # Plot model fit of mean out of pocket medical spending by age-income
        for i in range(5):
            plt.plot(AgeVec,X[11][i,:], '-' + income_colors[i])
        for i in range(5):
            plt.plot(AgeVec,Data.OOPbyIncAge[i,:], '.' + income_colors[i])
        plt.title('Mean out-of-pocket medical spending by income')
        plt.ylabel(r'OOP expenses $o_{it}$, \$10,000 USD (y2000)')
        plt.xlabel('Age')
        plt.legend(['Bottom quintile','Second quintile','Third quintile','Fourth quintile','Top quintile'],loc=2)
        if save_figs:
            plt.savefig('./Figures/OOPbyIncAge.pdf')
        plt.show()
    
        # Plot model fit of stdev out of pocket medical spending by age
        plt.plot(AgeVec,X[1],'-k')
        plt.plot(AgeVec,Data.StDevOOPbyAge,'.k')
        plt.title('Standard deviation of out-of-pocket medical spending [4(a)]')
        plt.ylabel(r'OOP expenses $o_{it}$, \$10,000 USD (y2000)')
        plt.xlabel('Age')
        if save_figs:
            plt.savefig('./Figures/StDevOOPbyAge.pdf')
        plt.show()
        
        # Plot model fit of stdev out of pocket medical spending by age and health
        for h in range(3):
            plt.plot(AgeVec,X[4][h,:], '-' + health_colors[h])
        for h in range(3):
            plt.plot(AgeVec,Data.StDevOOPbyHealthAge[h,:], '.' + health_colors[h])
        plt.legend(['Bottom tertile','Middle tertile','Top tertile'],loc=2)
        plt.title('Standard deviation of out-of-pocket medical spending by health [4(b)]')
        plt.ylabel(r'OOP expenses $o_{it}$, \$10,000 USD (y2000)')
        plt.xlabel('Age')
        if save_figs:
            plt.savefig('./Figures/StDevOOPbyHealthAge.pdf')
        plt.show()
        
        # Plot model fit of mortality by age
        plt.plot(AgeVec, X[2],'-k')
        plt.plot(AgeVec, Data.MortByAge,'.k')
        plt.title('Mortality probability [5(a)]')
        plt.ylabel(r'Probability $D_{it}$')
        plt.xlabel('Age')
        if save_figs:
            plt.savefig('./Figures/MortByAge.pdf')
        plt.show()
        
        # Plot model fit of mortality by age and health for females
        for h in range(3):
            plt.plot(AgeVec, X[8][0,h,:], '-' + health_colors[h])
        for h in range(3):
            plt.plot(AgeVec, Data.MortBySexHealthAge[0,h,:], '.' + health_colors[h])
        plt.title('Mortality probability by health, women [5(c).1]')
        plt.ylabel(r'Probability $D_{it}$')
        plt.xlabel('Age')
        plt.legend(['Bottom tertile','Middle tertile','Top tertile'],loc=2)
        if save_figs:
            plt.savefig('./Figures/MortByHealthAgeWomen.pdf')
        plt.show()
        
        # Plot model fit of mortality by age and health for males
        for h in range(3):
            plt.plot(AgeVec, X[8][1,h,:], '-' + health_colors[h])
        for h in range(3):
            plt.plot(AgeVec, Data.MortBySexHealthAge[1,h,:], '.' + health_colors[h])
        plt.title('Mortality probability by health, men [5(c).2]')
        plt.ylabel(r'Probability $D_{it}$')
        plt.xlabel('Age')
        plt.legend(['Bottom tertile','Middle tertile','Top tertile'],loc=2)
        if save_figs:
            plt.savefig('./Figures/MortByHealthAgeMen.pdf')
        plt.show()
        
        # Plot model fit of mortality by age and health quintile
        for h in range(5):
            plt.plot(AgeVec, X[17][h,:], '-' + income_colors[h])
        for h in range(5):
            plt.plot(AgeVec, Data.MortByHealthAge[h,:], '.' + income_colors[h])
        plt.title('Mortality probability by health [5(b)]')
        plt.ylabel(r'Probability $D_{it}$')
        plt.xlabel('Age')
        plt.legend(['Bottom quintile','Second quintile','Third quintile','Fourth quintile','Top quintile'],loc=2)
        if save_figs:
            plt.savefig('./Figures/MortByHealthAge.pdf')
        plt.show()
    
        # Plot model fit of wealth by age and income quintile
        for i in range(5):
            plt.plot(AgeVec, X[9][i,:], '-' + income_colors[i])
        for i in range(5):
            plt.plot(AgeVec, Data.WealthByIncAge[i,:], '.' + income_colors[i])
        plt.title('Median wealth by income quintile [1(a)]')
        plt.ylabel(r'Assets $a_{it}$, \$10,000 USD (y2000)')
        plt.xlabel('Age')
        if save_figs:
            plt.savefig('./Figures/WealthByIncAge.pdf')
        plt.show()
        
        # Plot model fit of wealth by age and wealth quintile (for one income quintile at a time)
        names = ['bottom','second','third','fourth','top']
        for i in range(5):
            for j in range(5):
                plt.plot(AgeVec, X[12][i,j,:], '-' + income_colors[j])
            for j in range(5):
                plt.plot(AgeVec, Data.WealthByIncWealthAge[i,j,:], '.' + income_colors[j])
            plt.title('Median wealth by wealth quintile: ' + names[i] + ' income quintile')
            plt.ylabel(r'Assets $a_{it}$, \$10,000 USD (y2000)')
            plt.xlabel('Age')
            if save_figs:
                plt.savefig('./Figures/WealthByIncWealthAge' + str(i+1) + '.pdf')
            plt.show()
            
        # Plot model fit of wealth by age and health quintile
        for i in range(5):
            plt.plot(AgeVec, X[19][i,:], '-' + income_colors[i])
        for i in range(5):
            plt.plot(AgeVec, Data.WealthByHealthAge[i,:], '.' + income_colors[i])
        plt.title('Median wealth by health quintile')
        plt.ylabel(r'Assets $a_{it}$, \$10,000 USD (y2000)')
        plt.xlabel('Age')
        plt.ylim([-1.,19.])
        plt.legend(['Bottom quintile','Second quintile','Third quintile','Fourth quintile','Top quintile'],loc=1)
        if save_figs:
            plt.savefig('./Figures/WealthByHealthAge.pdf')
        plt.show()
        
        
        # Plot model fit of mean health by health, sex, and age
        for h in range(3):
            plt.plot(AgeVec, X[6][0,h,:], '-' + health_colors[h])
        for h in range(3):
            plt.plot(AgeVec, Data.HealthBySexHealthAge[0,h,:], '.' + health_colors[h])
        plt.title('Mean health by health tertile, women [2(a).1]')
        plt.ylabel(r'Health capital $h_{it}$')
        plt.xlabel('Age')
        plt.ylim([0.2,0.9])
        plt.legend(['Bottom tertile','Middle tertile','Top tertile'],loc=1)
        if save_figs:
            plt.savefig('./Figures/HealthbyHealthAgeWomen.pdf')
        plt.show()
        
        for h in range(3):
            plt.plot(AgeVec, X[6][1,h,:], '-' + health_colors[h])
        for h in range(3):
            plt.plot(AgeVec, Data.HealthBySexHealthAge[1,h,:], '.' + health_colors[h])
        plt.title('Mean health by health tertile, men [2(a).2]')
        plt.ylabel(r'Health capital $h_{it}$')
        plt.xlabel('Age')
        plt.ylim([0.2,0.9])
        plt.legend(['Bottom tertile','Middle tertile','Top tertile'],loc=1)
        if save_figs:
            plt.savefig('./Figures/HealthbyHealthAgeMen.pdf')
        plt.show()
        
        # Plot model fit of mean health by health and age
        for h in range(5):
            plt.plot(AgeVec, X[18][h,:], '-' + income_colors[h])
        for h in range(5):
            plt.plot(AgeVec, Data.HealthByHealthAge[h,:], '.' + income_colors[h])
        plt.legend(['Bottom quintile','Second quintile','Third quintile','Fourth quintile','Top quintile'],loc=1)
        plt.title('Mean health by health quintile [2(c)]')
        plt.ylabel(r'Health capital $h_{it}$')
        plt.xlabel('Age')
        plt.ylim([0.2,0.9])
        if save_figs:
            plt.savefig('./Figures/HealthbyHealthAge.pdf')
        plt.show()
        
        # Plot model fit of mean health by health and age
        for i in range(5):
            plt.plot(AgeVec, X[10][i,:], '-' + income_colors[i])
        for i in range(5):
            plt.plot(AgeVec, Data.HealthByIncAge[i,:], '.' + income_colors[i])
        plt.legend(['Bottom quintile','Second quintile','Third quintile','Fourth quintile','Top quintile'],loc=1)
        plt.title('Mean health by income quintile, [2(b)]')
        plt.ylabel(r'Health capital $h_{it}$')
        plt.xlabel('Age')
        plt.ylim([0.39,0.75])
        if save_figs:
            plt.savefig('./Figures/HealthbyIncAge.pdf')
        plt.show()
        
        # Plot model fit of "health next coefficient" by wealth and income
        for i in range(5):
            plt.plot(np.arange(1,6),X[15][i,:], income_colors[i] + '-')
        for i in range(5):
            plt.plot(np.arange(1,6),Data.AvgHealthResidualByIncWealth[i,:], '.' + income_colors[i])
        plt.xlabel('Wealth quintile')
        plt.ylabel(r'Coeff on income-wealth quintile $\gimel_d[k,\ell]$, \$10k')
        plt.xticks(np.arange(1,6),['Bottom','Second','Third','Fourth','Top'])
        plt.title('Average health produced by income and wealth quintile [7(a)]')
        if save_figs:
            plt.savefig('./Figures/HealthCoeffByIncWealth.pdf')
        plt.show()
        
        # Plot model fit of standard deviation of change in health by age
        plt.plot(AgeVec, X[3], '-k')
        plt.plot(AgeVec, Data.StDevDeltaHealthByAge, '.k')
        plt.title('Standard deviation of change in health [6(a)]')
        plt.ylabel(r'Standard deviation of $\Delta h_{it}$')
        plt.xlabel('Age')
        if save_figs:
            plt.savefig('./Figures/StDevDeltaHealthByAge.pdf')
        plt.show()
        
        # Plot model fit of standard deviation of change in health by age and health
        for h in range(3):
            plt.plot(AgeVec, X[5][h,:], '-' + health_colors[h])
        for h in range(3):
            plt.plot(AgeVec, Data.StDevDeltaHealthByHealthAge[h,:], '.' + health_colors[h])
        plt.title('Standard deviation of change in health by health tertile [6(b)]')
        plt.ylabel(r'Standard deviation of $\Delta h_{it}$')
        plt.xlabel('Age')
        plt.legend(['Bottom tertile','Middle tertile','Top tertile'],loc=4)
        if save_figs:
            plt.savefig('./Figures/StDevDeltaHealthByHealthAge.pdf')
        plt.show()
    


    if perturb_one_param:
        # Test model identification by perturbing one parameter at a time
        param_i = 25
        param_min = -2.2
        param_max = -2.1
        N = 21
        perturb_vec = np.linspace(param_min,param_max,num=N)
        fit_vec = np.zeros(N) + np.nan
        for j in range(N):
            params = copy(Params.test_param_vec)
            params[param_i] = perturb_vec[j]
            fit_vec[j] = objectiveFunctionWrapper(params)
            
        plt.plot(perturb_vec,fit_vec)
        plt.xlabel(Params.param_names[param_i])
        plt.ylabel('Sum of squared moment differences')
        plt.show()


    if perturb_two_params:
        # Test model identification by perturbing two parameters at a time
        import pylab
        level_count = 100
        param1_i = 31
        param2_i = 32
        param1_min = -2.0
        param1_max = -1.6
        param2_min = -0.4
        param2_max = 0.0
        N = 20
        param1_vec = np.linspace(param1_min,param1_max,N)
        param2_vec = np.linspace(param2_min,param2_max,N)
        param1_mesh, param2_mesh = pylab.meshgrid(param1_vec,param2_vec)
        fit_array = np.zeros([N,N]) + np.nan
        for i in range(N):
            param1 = param1_vec[i]
            for j in range(N):
                param2 = param2_vec[j]
                params = copy(Params.test_param_vec)
                params[param1_i] = param1
                params[param2_i] = param2
                fit_array[i,j] = objectiveFunctionWrapper(params)
                print(i,j,fit_array[i,j])
        smm_contour = pylab.contourf(param2_mesh,param1_mesh,fit_array.transpose()**0.1,40)
        pylab.colorbar(smm_contour)
        pylab.xlabel(Params.param_names[param2_i])
        pylab.ylabel(Params.param_names[param1_i])
        pylab.show()
    
    

    if estimate_model:
        # Estimate some (or all) of the model parameters
        which_indices = which_indices = np.array([0,1,2,3,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32])
        #which_indices = np.array([0,1,5,6,7])
        which_bool = np.zeros(33,dtype=bool)
        which_bool[which_indices] = True
        estimated_params = minimizeNelderMead(objectiveFunctionWrapper,Params.test_param_vec,verbose=True,which_vars=which_bool)
        for i in which_indices.tolist():
            print(Params.param_names[i] + ' = ' + str(estimated_params[i]))
    


    if calc_std_errs:
        # Calculate standard errors using Jacobian of moment difference function for some or all parameters
        which_indices = np.array([0,1,2,3,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,25,26,27,28,29,30,31,32])
        #which_indices = np.array([0,1,5,6,7])
        which_bool = np.zeros(33,dtype=bool)
        which_bool[which_indices] = True
        standard_errors, cov_matrix, moment_derivatives = calcStdErrs(Params.test_param_vec,Data.use_cohorts,which_bool,eps=0.001)
        for n in range(which_indices.size):
            i = which_indices[n]
            print(Params.param_names[i] + ' = ' + str(standard_errors[n]))
        for n in range(which_indices.size):
            for nn in range(n):
                corr = cov_matrix[n,nn]/(standard_errors[n]*standard_errors[nn])
                if np.abs(corr > 0.7):
                    print(Params.param_names[which_indices[n]] + ' and ' + Params.param_names[which_indices[nn]] + ' have a correlation of ' + str(corr))
        stderrs_adj = np.zeros(33) + np.nan
        stderrs_adj[which_bool] = standard_errors
        makeParamTable('EstimatedParameters', Params.test_param_vec, stderrs=stderrs_adj)
        temp = np.dot(moment_derivatives,MomentWeights)
        sensitivity = -np.dot(np.linalg.inv(np.dot(temp,moment_derivatives.transpose())),temp)
        
        
    if calc_std_errs_alt:
        # Calculate standard errors using approximation of Hessian of objective function for some or all parameters
        which_indices = np.array([0,1,5,6,7])
        which_bool = np.zeros(33,dtype=bool)
        which_bool[which_indices] = True
        standard_errors, cov_matrix = calcStdErrsAlt(Params.test_param_vec,Data.use_cohorts,which_bool,f_dev=5.)
        for n in range(which_indices.size):
            i = which_indices[n]
            print(Params.param_names[i] + ' = ' + str(standard_errors[n]))
        for n in range(which_indices.size):
            for nn in range(n):
                corr = cov_matrix[n,nn]/(standard_errors[n]*standard_errors[nn])
                if np.abs(corr > 0.7):
                    print(Params.param_names[which_indices[n]] + ' and ' + Params.param_names[which_indices[nn]] + ' have a correlation of ' + str(corr))
        #makeParamTable('EstimatedParameters',Params.test_param_vec[which_indices],which_indices,stderrs=standard_errors)
