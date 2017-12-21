'''
This module runs the structural estimation for the Ounce of Prevention project.
'''

import sys
import os
sys.path.insert(0,'../')

from time import clock
from copy import copy
import numpy as np
from HARKparallel import multiThreadCommands
from HARKestimation import minimizeNelderMead
from HARKutilities import getPercentiles
from HealthInvModel import HealthInvestmentConsumerType
import LoadHealthInvData as Data
import HealthInvParams as Params
import matplotlib.pyplot as plt

# Import objects from the data loading module
DataMoments = Data.all_moments
MomentWeights = Data.weighting_matrix
MomentMask = Data.moment_mask
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
        ThisType.track_vars = ['OOPmedNow','hLvlNow','aLvlNow','CumLivPrb','DiePrbNow']
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
#    T = type_list[0].T_sim
#   N = aLvlHist.shape[1]
    
    # Combine data labels across types
    HealthTert = np.concatenate([this_type.HealthTert for this_type in type_list])
    WealthQuint = np.concatenate([this_type.WealthQuint for this_type in type_list])
    IncQuint = np.concatenate([this_type.IncQuintLong for this_type in type_list])
    Sex = np.concatenate([this_type.SexLong for this_type in type_list])
    
    # Combine in-data-span masking array across all types
    InDataSpan = np.concatenate([this_type.InDataSpanArray for this_type in type_list],axis=1)
    
    # Determine eligibility to be used for various purposes
    Active = hLvlHist > 0.
#    AliveNextPeriod = np.zeros((T,N))
#    AliveNextPeriod[:-1,:] = Alive[1:,:]
#    DeadNextPeriod = np.logical_not(AliveNextPeriod)
#    AliveNowAndLater = np.logical_and(Alive,AliveNextPeriod)
#    DyingThisPeriod = np.logical_and(Alive,DeadNextPeriod)
    
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
                MeanOOP = np.dot(OOP,Weight)/WeightSum
                OOPbySexHealthAge[s,h,t] = MeanOOP
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
    MomentDerivativeArray = np.zeros((N,1770))
    
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
    return StdErrVec


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
        MomentDifferences = np.reshape((SimulatedMoments - DataMoments)*MomentMask,(1770,1))
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
        'HealthProd0' : np.exp(param_vec[24]),
        'HealthProd1' : np.exp(param_vec[25]),
        'HealthProd2' : param_vec[26],
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
    weighted_moment_sum = objectiveFunction(these_params,use_cohorts,True)
    
    # Write the current parameters to a file if a certain number of function calls have happened
    Params.func_call_count += 1
    if np.mod(Params.func_call_count,Params.store_freq) == 0:
        writeParamsToFile(param_vec,'ParameterStatus.txt')
    
    return weighted_moment_sum



if __name__ == '__main__':

#    param_dict = convertVecToDict(Params.test_param_vec)
#    MyTypes = makeMultiTypeSimple(param_dict)
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
    plt.ylabel('Mean OOP medical spending, women')
    plt.show()
    
    # Plot model fit of mean out of pocket medical spending by age-health for males
    plt.plot(X[7][1,:,:].transpose())
    for h in range(3):
        plt.plot(Data.OOPbySexHealthAge[1,h,:],'--')
    plt.ylabel('Mean OOP medical spending, men')
    plt.show()
    
    # Plot model fit of mean out of pocket medical spending by age-income
    plt.plot(X[11].transpose())
    plt.plot(Data.OOPbyIncAge.transpose(),'.')
    plt.ylabel('Mean OOP medical spending by income quintile')
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
    


    
#    # Test model identification by perturbing one parameter at a time
#    param_i = 16
#    param_min = -0.01
#    param_max = 0.02
#    N = 30
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



#    # Test model identification by perturbing two parameters at a time
#    import pylab
#    level_count = 100
#    param1_i = 31
#    param2_i = 32
#    param1_min = -2.0
#    param1_max = -1.6
#    param2_min = -0.4
#    param2_max = 0.0
#    N = 20
#    param1_vec = np.linspace(param1_min,param1_max,N)
#    param2_vec = np.linspace(param2_min,param2_max,N)
#    param1_mesh, param2_mesh = pylab.meshgrid(param1_vec,param2_vec)
#    fit_array = np.zeros([N,N]) + np.nan
#    for i in range(N):
#        param1 = param1_vec[i]
#        for j in range(N):
#            param2 = param2_vec[j]
#            params = copy(Params.test_param_vec)
#            params[param1_i] = param1
#            params[param2_i] = param2
#            fit_array[i,j] = objectiveFunctionWrapper(params)
#            print(i,j,fit_array[i,j])
#    smm_contour = pylab.contourf(param2_mesh,param1_mesh,fit_array.transpose()**0.1,40)
#    pylab.colorbar(smm_contour)
#    pylab.xlabel(Params.param_names[param2_i])
#    pylab.ylabel(Params.param_names[param1_i])
#    pylab.show()
#    
    


#    # Estimate some (or all) of the model parameters
#    which_indices = np.array([0,1,2,5,6,7,8,9,10,11,12,13,14,15])
#    which_bool = np.zeros(33,dtype=bool)
#    which_bool[which_indices] = True
#    estimated_params = minimizeNelderMead(objectiveFunctionWrapper,Params.test_param_vec,verbose=True,which_vars=which_bool)
#    for i in which_indices.tolist():
#        print(Params.param_names[i] + ' = ' + str(estimated_params[i]))
    



#    # Calculate standard errors for some or all parameters
#    which_indices = np.array([0,1,2,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,27,28,29,30,31,32])
#    which_bool = np.zeros(33,dtype=bool)
#    which_bool[which_indices] = True
#    standard_errors = calcStdErrs(Params.test_param_vec,Data.use_cohorts,which_bool,eps=0.001)
#    for n in range(which_indices.size):
#        i = which_indices[n]
#        print(Params.param_names[i] + ' = ' + str(standard_errors[n]))
#        