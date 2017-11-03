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
        
    multiThreadCommands(type_list,['estimationAction()'])
    return type_list



if __name__ == '__main__':

    #MyTypes = makeMultiTypeSimple(Params.test_params)
    #t_start = clock()
    #MyTypes[0].estimationAction()
    #t_end = clock()
    #print('Processing one agent type took ' + str(t_end-t_start) + ' seconds.')
    
    t_start = clock()
    MyTypes = processSimulatedTypes(Params.test_params,False)
    t_end = clock()
    print('Processing ten agent types took ' + str(t_end-t_start) + ' seconds.')
    
#    t=0
#    bMax=200.
#    
#    for j in range(10):
#        MyTypes[j].plotxFuncByHealth(t,MedShk=1.0,bMax=bMax)
    