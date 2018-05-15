'''
This module runs the structural estimation for the Ounce of Prevention project.
'''

import sys
import os
sys.path.insert(0,'../')

from time import clock
from copy import copy
import numpy as np
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
        self.T_sim = self.T_cycle
        self.AgentCount = X*self.DataAgentCount
        self.WealthQuint = np.tile(self.WealthQuint,X)
        self.HealthTert = np.tile(self.HealthTert,X)
        self.aLvlInit = np.tile(self.aLvlInit,X)
        self.HlvlInit = np.tile(self.HlvlInit,X)
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
        tempw = 2. - np.exp(self.LogJerk)
        HealthProd0 = (tempw-1.)/(tempw-2.)
        tempx = np.exp(self.LogSlope) # Slope of health production function at iLvl=0
        tempy = -np.exp(self.LogSlope+self.LogCurve) # Curvature of health prod at iLvl=0
        HealthProd2 = tempx/tempy*(HealthProd0-1.)
        HealthProd1 = tempx/HealthProd0*HealthProd2**(1.-HealthProd0)
        if tempx > 0.:
            HealthProdFunc = lambda i : tempx/HealthProd0*((i*HealthProd2**((1.-HealthProd0)/HealthProd0) + HealthProd2**(1./HealthProd0))**HealthProd0 - HealthProd2)
            MargHealthProdInvFunc = lambda q : HealthProd2*((q/tempx)**(1./(HealthProd0-1.)) -1.)
        else:
            HealthProdFunc = lambda i : 0.*i
            MargHealthProdInvFunc = lambda q : 0.*q
        
        # Define the (marginal)(inverse) health production function
        self.HealthProdFunc = HealthProdFunc
        self.MargHealthProdInvFunc = MargHealthProdInvFunc
        self.addToTimeInv('HealthProdFunc','MargHealthProdInvFunc')
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
        ThisType.aLvlInit = Data.w_init[these]
        ThisType.HlvlInit = Data.h_init[these]
        ThisType.BornBoolArray = Data.BornBoolArray[:,these]
        ThisType.InDataSpanArray = Data.InDataSpanArray[:,these]
        ThisType.track_vars = ['OOPmedNow','hLvlNow','aLvlNow','CumLivPrb','DiePrbNow','RatioNow','MedLvlNow','CopayNow']
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
        
    multiThreadCommands(type_list,['estimationAction()'],num_jobs=10)
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
                AvgOOPResidualByIncWealth
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
                AvgOOPResidualByIncWealth.flatten()
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
    CopayHist = np.concatenate([this_type.CopayNow_hist for this_type in type_list],axis=1)
    
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
    Copay_reg = CopayHist[THESE]
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
        OOP_reg = (HealthInv + Med_reg)*Copay_reg
        
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
    The objective function for the ounce of prevention estimation.  Takes a dictionary
    of parameters and a boolean indicator for whether to break sample up by cohorts.
    Returns a single real representing the weighted sum of squared moment distances.
    
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
    return StdErrVec, ParamCovMatrix


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
    
    if False:
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
#    # Translate the parameter vector 
#    HealthProd0 = param_vec[24]   # "Ultimate" curvature of health production function
#    tempx = np.exp(param_vec[25]) # Slope of health production function at iLvl=0
#    tempy = -np.exp(param_vec[25]+param_vec[26]) # Curvature of health prod at iLvl=0
#    if tempx > 0.:
#        HealthProd2 = tempx/tempy*(HealthProd0-1.)
#        HealthProd1 = tempx/HealthProd0*HealthProd2**(1.-HealthProd0)
#    else:
#        HealthProd2 = 1.
#        HealthProd1 = 0.
    
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
#    MyTypes[i].plotxFuncByHealth(t,MedShk=0.1,bMax=bMax)
#    MyTypes[i].plotxFuncByMedShk(t,hLvl=0.9,bMax=bMax)
#    MyTypes[i].plotiFuncByHealth(t,MedShk=0.1,bMax=bMax)
#    MyTypes[i].plotiFuncByMedShk(t,hLvl=0.9,bMax=bMax)
#    MyTypes[i].plotvFuncByHealth(t,bMax=bMax)
#    MyTypes[i].plotdvdbFuncByHealth(t,bMax=bMax)
#    MyTypes[i].plotdvdhFuncByHealth(t,bMax=bMax)
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
    test_obj_func = False
    plot_model_fit = False
    perturb_one_param = False
    perturb_two_params = False
    estimate_model = False
    calc_std_errs = False


    if test_obj_func:
        t_start = clock()
        X = objectiveFunctionWrapper(Params.test_param_vec)
        t_end = clock()
        print('One objective function evaluation took ' + str(t_end-t_start) + ' seconds.')
    
    if plot_model_fit:
        # Plot model fit of mean out of pocket medical spending by age
        plt.plot(X[0])
        plt.plot(Data.OOPbyAge,'.k')
        plt.ylabel('Mean OOP medical spending')
        plt.show()
        
        # Plot model fit of mean out of pocket medical spending by age-health for all
        plt.plot(X[7][0,:,:].transpose())
        for h in range(3):
            plt.plot(Data.OOPbySexHealthAge[0,h,:],'--')
        plt.ylabel('Mean OOP medical spending by health tertile')
        plt.show()
        
        ## Plot model fit of mean out of pocket medical spending by age-health for males
        #plt.plot(X[7][1,:,:].transpose())
        #for h in range(3):
        #    plt.plot(Data.OOPbySexHealthAge[1,h,:],'--')
        #plt.ylabel('Mean OOP medical spending, men')
        #plt.show()
        
        # Plot model fit of "OOP coefficient" by wealth and income
        plt.plot(X[16].transpose())
        plt.plot(Data.AvgOOPResidualByIncWealth.transpose(),'--')
        plt.xlabel('Wealth quintile')
        plt.ylabel('OOP coefficient by income quintile')
        plt.show()
        
        ## Plot model fit of mean out of pocket medical spending by age-income
        #plt.plot(X[11].transpose())
        #plt.plot(Data.OOPbyIncAge.transpose(),'.')
        #plt.ylabel('Mean OOP medical spending by income quintile')
        #plt.show()
    
        # Plot model fit of stdev out of pocket medical spending by age
        plt.plot(X[1])
        plt.plot(Data.StDevOOPbyAge,'.k')
        plt.ylabel('StDev OOP medical spending')
        plt.show()
        
        # Plot model fit of stdev out of pocket medical spending by age and health
        plt.plot(X[4].transpose())
        for h in range(3):
            plt.plot(Data.StDevOOPbyHealthAge[h,:],'--')
        plt.ylabel('StDev OOP medical spending')
        plt.show()
        
        # Plot model fit of mortality by age
        plt.plot(X[2])
        plt.plot(Data.MortByAge,'.k')
        plt.ylabel('Mortality probability')
        plt.show()
        
        # Plot model fit of mortality by age and health for females
        plt.plot(X[8][0,:,:].transpose())
        for h in range(3):
            plt.plot(Data.MortBySexHealthAge[0,h,:],'.')
        plt.ylabel('Mortality probability, women')
        plt.show()
        
        # Plot model fit of mortality by age and health for males
        plt.plot(X[8][1,:,:].transpose())
        for h in range(3):
            plt.plot(Data.MortBySexHealthAge[1,h,:],'.')
        plt.ylabel('Mortality probability, men')
        plt.show()
    
        # Plot model fit of wealth by age and income quintile
        plt.plot(X[9].transpose())
        for i in range(5):
            plt.plot(Data.WealthByIncAge[i,:],'.')
        plt.ylabel('Median wealth profiles')
        plt.show()
        
        # Plot model fit of wealth by age and wealth quintile (for one income quintile at a time)
        names = ['lowest','second','third','fourth','highest']
        for i in range(5):
            plt.plot(X[12][i,:,:].transpose())
            for j in range(5):
                plt.plot(Data.WealthByIncWealthAge[i,j,:],'.')
            plt.ylabel('Median wealth profiles for ' + names[i] + ' income quintile')
            plt.show()
        
        # Plot model fit of mean health by health, sex, and age
        plt.plot(X[6][0,:,:].transpose())
        plt.plot(Data.HealthBySexHealthAge[0,:,:].transpose(),'.k')
        plt.ylabel('Health profiles by health tertile, women')
        plt.show()
        plt.plot(X[6][1,:,:].transpose())
        plt.plot(Data.HealthBySexHealthAge[1,:,:].transpose(),'.k')
        plt.ylabel('Health profiles by health tertile, men')
        plt.show()
        
        # Plot model fit of mean health by income and age
        plt.plot(X[10].transpose())
        plt.plot(Data.HealthByIncAge.transpose(),'--')
        plt.ylabel('Health profiles by income quintile')
        plt.show()
        
        # Plot model fit of "health next coefficient" by wealth and income
        plt.plot(X[15].transpose())
        plt.plot(Data.AvgHealthResidualByIncWealth.transpose(),'--')
        plt.xlabel('Wealth quintile')
        plt.ylabel('Health next coefficient by income quintile')
        plt.show()
        
        ## Plot model fit of mean health by age and wealth quintile (for one income quintile at a time)
        #names = ['lowest','second','third','fourth','highest']
        #for i in range(5):
        #    plt.plot(X[13][i,:,:].transpose())
        #    for j in range(5):
        #        plt.plot(Data.HealthByIncWealthAge[i,j,:],'.')
        #    plt.ylabel('Health profiles for ' + names[i] + ' income quintile')
        #    plt.show()
        
        # Plot model fit of standard deviation of change in health by age
        plt.plot(X[3])
        plt.plot(Data.StDevDeltaHealthByAge,'.k')
        plt.ylabel('Standard deviation of change in health')
        plt.show()
        
        # Plot model fit of standard deviation of change in health by age and health
        plt.plot(X[5].transpose())
        plt.plot(Data.StDevDeltaHealthByHealthAge.transpose(),'.')
        plt.ylabel('Standard deviation of change in health')
        plt.show()
    


    if perturb_one_param:
        # Test model identification by perturbing one parameter at a time
        param_i = 2
        param_min = 3.0
        param_max = 12.0
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
        which_indices = np.array([0,1,5,6,7])
        which_bool = np.zeros(33,dtype=bool)
        which_bool[which_indices] = True
        estimated_params = minimizeNelderMead(objectiveFunctionWrapper,Params.test_param_vec,verbose=True,which_vars=which_bool)
        for i in which_indices.tolist():
            print(Params.param_names[i] + ' = ' + str(estimated_params[i]))
    


    if calc_std_errs:
        # Calculate standard errors for some or all parameters
        which_indices = np.array([8,10,11,12,13,14,15])
        which_bool = np.zeros(33,dtype=bool)
        which_bool[which_indices] = True
        standard_errors, cov_matrix = calcStdErrs(Params.test_param_vec,Data.use_cohorts,which_bool,eps=0.001)
        for n in range(which_indices.size):
            i = which_indices[n]
            print(Params.param_names[i] + ' = ' + str(standard_errors[n]))
        makeParamTable('EstimatedParameters',Params.test_param_vec[which_indices],which_indices,stderrs=standard_errors)
        
        