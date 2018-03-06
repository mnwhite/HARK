'''
This module sets up and executes counterfactual policy exercises for the Ounce of Prevention project.
'''

import sys
import os
sys.path.insert(0,'../')

from time import clock
from copy import copy
import numpy as np
import LoadHealthInvData as Data
from HARKcore import HARKobject
from HARKutilities import getPercentiles
from HealthInvEstimation import convertVecToDict, EstimationAgentType

# Define a class for representing counterfactual subsidy policies
class SubsidyPolicy(HARKobject):
    '''
    A class for representing subsidy policies for health investment.  All of the
    attributes listed in policy_attributes should be passed to the constructor
    method as lists of the same length as the list of AgentTypes that will be
    used in the counterfactual exercises.
    '''
    policy_attributes = ['Subsidy0','Subsidy1']
    
    def __init__(self, **kwds):
        for key in kwds:
            setattr(self,key,kwds[key])
            
    
    def enactPolicy(self,Agents):
        '''
        Apply the counterfactual policy to all of the types in a list of agents.
        
        Parameters
        ----------
        Agents : [AgentType]
            List of types of consumers in the economy.
            
        Returns
        -------
        None
        '''
        for name in self.policy:
            for i in range(len(Agents)):
                setattr(Agents[i],name,getattr(self,name)[i])
                
                
def makeMultiTypeCounterfactual(params):
    '''
    Create 10 instances of the estimation agent type by splitting respondents
    by sex-income quintile.  Passes the parameter dictionary to each one. This
    version is for the counterfactuals, so the data agents are from cohorts 16-18.
    
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
        
        these = Data.TypeBoolArrayEstimation[n,:]
        ThisType.DataAgentCount = np.sum(these)
        ThisType.WealthQuint = Data.wealth_quint_data[these]
        ThisType.HealthTert = Data.health_tert_data[these]
        ThisType.aLvlInit = Data.w_init[these] # These need to switch to 2010 data
        ThisType.HlvlInit = Data.h_init[these]
        ThisType.track_vars = ['OOPmedNow','hLvlNow','aLvlNow','CumLivPrb','DiePrbNow','RatioNow','MedLvlNow','CopayNow']
        ThisType.seed = n
        
        type_list.append(ThisType)
        
    return type_list
                
                
                
# Define a function for running a set of counterfactuals
def runCounterfactuals(name,Parameters,Policies):
    '''
    Run a set of counterfactual policies given a set of parameters.
    
    Parameters
    ----------
    name : str
        Name of this counterfactual set, used in filenames.
    Parameters : np.array
        A size 33 array of parameters, just like for the estimation.
    Policies : [SubsidyPolicy]
        List of counterfactual policies to simulate.
    
    Returns
    -------
    TBD
    '''
    # Make the agent types and have them create extra functions during solution
    param_dict = convertVecToDict(Parameters)
    Agents = makeMultiTypeCounterfactual(param_dict)
    for agent in Agents:
        agent(CalcExpectationFuncs = True)
            
            

if __name__ == '__main__':
    pass