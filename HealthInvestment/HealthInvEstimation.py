'''
This module runs the structural estimation for the Ounce of Prevention project.
'''

import sys
import os
sys.path.insert(0,'../')

from time import clock
from copy import copy
import numpy as np
from HARKparallel import multiThreadCommands, multiThreadCommandsFake
from HealthInvModel import HealthInvestmentConsumerType
import LoadHealthInvData as Data
import HealthInvParams as Params
import matplotlib.pyplot as plt

DataMoments = Data.all_moments # Rename this to keep objects straight
Normalizer = Data.normalizer
CellSizes = Data.all_cell_sizes

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
        ThisType.track_vars = ['OOPmedNow','hLvlNow','aLvlNow']
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
        ThisType.track_vars = ['OOPmedNow','hLvlNow','aLvlNow']
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
    OOPhist  = np.concatenate([this_type.OOPmedNow_hist for this_type in type_list],axis=1)*10000
    T = type_list[0].T_sim
    N = aLvlHist.shape[1]
    
    # Combine data labels across types
    HealthTert = np.concatenate([this_type.HealthTert for this_type in type_list])
    WealthQuint = np.concatenate([this_type.WealthQuint for this_type in type_list])
    IncQuint = np.concatenate([this_type.IncQuintLong for this_type in type_list])
    Sex = np.concatenate([this_type.SexLong for this_type in type_list])
    
    # Combine in-data-span masking array across all types
    InDataSpan = np.concatenate([this_type.InDataSpanArray for this_type in type_list],axis=1)
    
    # Determine eligibility to be used for various purposes
    Alive = hLvlHist > 0.
    AliveNextPeriod = np.zeros((T,N))
    AliveNextPeriod[:-1,:] = Alive[1:,:]
    DeadNextPeriod = np.logical_not(AliveNextPeriod)
    AliveNowAndLater = np.logical_and(Alive,AliveNextPeriod)
    DyingThisPeriod = np.logical_and(Alive,DeadNextPeriod)
    
    # Calculate the change in health from period to period for all simulated agents
    DeltaHealth = np.zeros_like(hLvlHist)
    DeltaHealth[:-1,:] = hLvlHist[1:,:] - hLvlHist[:-1,:]
    
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
    
    # Loop through ages, sexes, quintiles, and health to fill in simulated moments
    for t in range(15):
        # Calculate mean and stdev of OOP medical spending, mortality rate, and stdev delta health by age
        THESE = np.logical_and(AliveNowAndLater[t,:],InDataSpan[t,:])
        OOP = OOPhist[t,THESE]
        OOPbyAge[t] = np.mean(OOP)
        StDevOOPbyAge[t] = np.std(OOP)
        NewDead = np.sum(np.logical_and(DyingThisPeriod[t,:],InDataSpan[t,:]))
        Total = NewDead + np.sum(THESE)
        MortByAge[t] = float(NewDead)/float(Total)
        StDevDeltaHealthByAge[t] = np.std(DeltaHealth[t,THESE])
        
        for h in range(3):
            # Calculate stdev OOP medical spending and stdev delta health by health by age
            right_health = HealthTert==(h+1)
            these = np.logical_and(THESE,right_health)
            OOP = OOPhist[t,these]
            StDevOOPbyHealthAge[h,t] = np.std(OOP)
            StDevDeltaHealthByHealthAge[h,t] = np.std(DeltaHealth[t,these])
            
            for s in range(2):
                # Calculate mean OOP medical spending and mortality by sex by health by age
                right_sex = Sex==s
                those = np.logical_and(these,right_sex)
                OOPbySexHealthAge[s,h,t] = np.mean(OOPhist[t,those])
                HealthBySexHealthAge[s,h,t] = np.mean(hLvlHist[t+1,those])
                NewDead = np.sum(np.logical_and(np.logical_and(np.logical_and(DyingThisPeriod[t,:],right_sex),right_health),InDataSpan[t,:]))
                Total = NewDead + np.sum(those)
                MortBySexHealthAge[s,h,t] = float(NewDead)/float(Total)
                
        for i in range(5):
            # Calculate median wealth, mean health, and mean OOP medical spending by income quintile by age
            right_inc = IncQuint == i+1
            these = np.logical_and(THESE,right_inc)
            WealthByIncAge[i,t] = np.median(aLvlHist[t,these])
            HealthByIncAge[i,t] = np.mean(hLvlHist[t+1,these])
            OOPbyIncAge[i,t] = np.mean(OOPhist[t,these])
            
            for j in range(5):
                # Calculate median wealth, mean health, and mean OOP medical spending by income quintile by wealth quintile by age
                right_wealth = WealthQuint == j+1
                those = np.logical_and(these,right_wealth)
                WealthByIncWealthAge[i,j,t] = np.median(aLvlHist[t,those])
                HealthByIncWealthAge[i,j,t] = np.mean(hLvlHist[t+1,those])
                OOPbyIncWealthAge[i,j,t] = np.mean(OOPhist[t,those])
                
    # Aggregate moments into a single vector and return
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
                OOPbyIncWealthAge
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
                OOPbyIncWealthAge.flatten()
                ])
    return all_moments


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
    
    if return_as_list:
        return SimulatedMoments
    else:
        MomentDifferences = (SimulatedMoments - DataMoments)/Normalizer
        MomentDifferencesSq = MomentDifferences**2
        weighted_moment_sum = np.dot(MomentDifferencesSq,CellSizes)
        return weighted_moment_sum


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
        'HealthProd0' : param_vec[24],
        'HealthProd1' : param_vec[25],
        'HealthProd2' : param_vec[26],
        'Mortality0' : param_vec[27],
        'MortalitySex' : param_vec[28],
        'MortalityAge' : param_vec[29],
        'MortalityAgeSq' : param_vec[30],
        'MortalityHealth' : param_vec[31],
        'MortalityHealthSq' : param_vec[32]
    }
    these_params = copy(Params.basic_estimation_dict)
    these_params.update(struct_params)
    
    # Run the objective function with the newly created dictionary
    use_cohorts = Data.use_cohorts
    weighted_moment_sum = objectiveFunction(these_params,use_cohorts,True)
    return weighted_moment_sum



if __name__ == '__main__':

#    MyTypes = makeMultiTypeSimple(Params.test_params)
#    t_start = clock()
#    MyTypes[0].estimationAction()
#    t_end = clock()
#    print('Processing one agent type took ' + str(t_end-t_start) + ' seconds.')
    
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



    t_start = clock()
    X = objectiveFunctionWrapper(Params.test_param_vec)
    t_end = clock()
    print('One objective function evaluation took ' + str(t_end-t_start) + ' seconds.')
    
    # Plot model fit of mean out of pocket medical spending by age
    plt.plot(X[0])
    plt.plot(Data.OOPbyAge,'.k')
    plt.ylabel('Mean OOP medical spending')
    plt.show()
    
    # Plot model fit of mean out of pocket medical spending by age-health for females
    plt.plot(X[7][0,:,:].transpose())
    for h in range(3):
        plt.plot(Data.OOPbySexHealthAge[0,h,:],'--')
    plt.ylabel('Mean OOP medical spending')
    plt.show()
    
    # Plot model fit of mean out of pocket medical spending by age-health for males
    plt.plot(X[7][1,:,:].transpose())
    for h in range(3):
        plt.plot(Data.OOPbySexHealthAge[1,h,:],'--')
    plt.ylabel('Mean OOP medical spending')
    plt.show()

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
    plt.plot(X[2].transpose())
    plt.plot(Data.MortByAge,'.k')
    plt.ylabel('Mortality probability')
    plt.show()
    
    # Plot model fit of mortality by age and health for females
    plt.plot(X[8][0,:,:].transpose())
    for h in range(3):
        plt.plot(Data.MortBySexHealthAge[0,h,:],'.')
    plt.ylabel('Mortality probability')
    plt.show()
    
    # Plot model fit of mortality by age and health for males
    plt.plot(X[8][1,:,:].transpose())
    for h in range(3):
        plt.plot(Data.MortBySexHealthAge[1,h,:],'.')
    plt.ylabel('Mortality probability')
    plt.show()

    # Plot model fit of wealth by age and income quintile
    plt.plot(X[9].transpose())
    for i in range(5):
        plt.plot(Data.WealthByIncAge[i,:],'.')
    plt.ylabel('Median wealth profiles')
    plt.show()
    


    
#    # Test model identification by perturbing one parameter at a time
#    param_i = 31
#    param_min = -1.7
#    param_max = -1.4
#    N = 10
#    perturb_vec = np.linspace(param_min,param_max,num=N)
#    fit_vec = np.zeros(N) + np.nan
#    for j in range(N):
#        params = copy(Params.test_param_vec)
#        params[param_i] = perturb_vec[j]
#        fit_vec[j] = objectiveFunctionWrapper(params)
#        
#    plt.plot(perturb_vec,fit_vec)
#    plt.xlabel(Params.param_names[param_i])
#    plt.ylabel('Sum of squared moment differences')
#    plt.show()
    