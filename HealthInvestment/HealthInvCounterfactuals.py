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
from HARKutilities import getPercentiles
from HARKcore import HARKobject
from HARKparallel import multiThreadCommands
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
                
                
# Define an agent class for the policy experiments, adding a few methods
class CounterfactualAgentType(EstimationAgentType):
    '''
    A class for representing agents in the Ounce of Prevention project, for the
    purpose of running counterfactual policy experiments.  Slightly extends the
    class EstimationAgentType, adding methods for counterfactuals.
    '''
    
    def runBaselineAction(self):
        '''
        Run methods for extracting relevant data from the baseline scenario.
        '''
        self.update()
        self.solve()
        self.repSimData()
        self.t_ageInit = self.BornBoolArray # other end of hacky fix
        self.evalExpectationFuncs()
        self.delSolution()
        
    def evalExpectationFuncs(self):
        '''
        Creates arrays with lifetime PDVs of several variables, from the perspective
        of the HRS subjects in 2010.  Stores arrays as attributes of self.
        '''
        # Initialize arrays that will be stored as attributes
        TotalMedPDVarray = np.nan*np.zeros(self.AgentCount)
        OOPmedPDVarray = np.nan*np.zeros(self.AgentCount)
        ExpectedLifeArray = np.nan*np.zeros(self.AgentCount)
        MedicarePDVarray = np.nan*np.zeros(self.AgentCount)
        SubsidyPDVarray = np.nan*np.zeros(self.AgentCount)
        WelfarePDVarray = np.nan*np.zeros(self.AgentCount)
        GovtPDVarray = np.nan*np.zeros(self.AgentCount)
        ValueArray = np.nan*np.zeros(self.AgentCount)
        
        # Loop through the three cohorts that are used in the counterfactual
        self.initializeSim()
        for t in range(3):
            # Select the agents with the correct starting age
            these = self.t_ageInit == t
            self.t_age = t
            self.t_sim = t
            self.ActiveNow[:] = False
            self.ActiveNow[these] = True
            
            # Advance the simulated agents into this period by simulating health shocks
            self.getShocks()
            self.getStates()
            
            # Evaluate the PDV functions (and value function), storing in the arrays
            bLvl = self.bLvlNow[these]
            hLvl = self.hLvlNow[these]
            TotalMedPDVarray[these] = self.solution[t].TotalMedPDVfunc(bLvl,hLvl)
            OOPmedPDVarray[these] = self.solution[t].OOPmedPDVfunc(bLvl,hLvl)
            ExpectedLifeArray[these] = self.solution[t].ExpectedLifeFunc(bLvl,hLvl)
            MedicarePDVarray[these] = self.solution[t].MedicarePDVfunc(bLvl,hLvl)
            SubsidyPDVarray[these] = self.solution[t].SubsidyPDVfunc(bLvl,hLvl)
            WelfarePDVarray[these] = self.solution[t].WelfarePDVfunc(bLvl,hLvl)
            GovtPDVarray[these] = self.solution[t].GovtPDVfunc(bLvl,hLvl)
            ValueArray[these] = self.solution[t].vFunc(bLvl,hLvl)
            
                
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
        ThisType = CounterfactualAgentType(**temp_dict)
        ThisType.IncomeNow = Data.IncomeArraySmall[n,:].tolist()
        ThisType.IncomeNext = Data.IncomeArraySmall[n,1:].tolist() + [1.]
        ThisType.addToTimeVary('IncomeNow','IncomeNext')
        ThisType.makeConstantMedPrice()
        ThisType.CohortNum = np.nan
        ThisType.IncQuint = np.mod(n,5)+1
        
        these = Data.TypeBoolArrayCounterfactual[n,:]
        ThisType.DataAgentCount = np.sum(these)
        ThisType.WealthQuint = Data.wealth_quint_data[these]
        ThisType.HealthTert = Data.health_tert_data[these]
        ThisType.aLvlInit = Data.w7_data[these]
        ThisType.HlvlInit = Data.h7_data[these]
        ThisType.t_ageInit = Data.age_in_2010[these]
        ThisType.BornBoolArray = ThisType.t_ageInit # hacky workaround
        ThisType.InDataSpanArray = np.zeros(10) # irrelevant
        ThisType.CalcExpectationFuncs = True
        ThisType.track_vars = ['OOPmedNow','hLvlNow','aLvlNow','CumLivPrb','DiePrbNow','RatioNow','MedLvlNow','CopayNow']
        ThisType.seed = n
        
        type_list.append(ThisType)
        
    return type_list
                


def calcSubpopStats(type_list):
    '''
    Calculate overall population averages and subpopulation averages for a 
    '''
                
                
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
    # Make the agent types
    param_dict = convertVecToDict(Parameters)
    Agents = makeMultiTypeCounterfactual(param_dict)
    
    # Solve the baseline model and get arrays of outcome variables and demographics
    multiThreadCommands(Agents,'runBaselineAction()')
    TotalMedBaseline, OOPmedBaseline, ExpectedLifeBaseline, MedicareBaseline, SubsidyBaseline, WelfareBaseline, GovtBaseline = calcSubpopStats(Agents)
    
    HealthTert = np.concatenate([this_type.HealthTert for this_type in Agents])
    WealthQuint = np.concatenate([this_type.WealthQuint for this_type in Agents])
    IncQuint = np.concatenate([this_type.IncQuintLong for this_type in Agents])
    Sex = np.concatenate([this_type.SexLong for this_type in Agents])
    TotalMedPDVarray = np.nan*np.zeros(self.AgentCount)
    OOPmedPDVarray = np.nan*np.zeros(self.AgentCount)
    ExpectedLifeArray = np.nan*np.zeros(self.AgentCount)
    MedicarePDVarray = np.nan*np.zeros(self.AgentCount)
    SubsidyPDVarray = np.nan*np.zeros(self.AgentCount)
    WelfarePDVarray = np.nan*np.zeros(self.AgentCount)
    GovtPDVarray = np.nan*np.zeros(self.AgentCount)
    ValueArray = np.nan*np.zeros(self.AgentCount)
    
            
            

if __name__ == '__main__':
    pass