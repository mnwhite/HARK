'''
This module sets up and executes counterfactual policy exercises for the Ounce of Prevention project.
'''

import sys
import os
sys.path.insert(0,'../')

from copy import copy, deepcopy
import numpy as np
from HARKutilities import getPercentiles
from HARKcore import HARKobject
from HARKparallel import multiThreadCommands, multiThreadCommandsFake
from HealthInvEstimation import convertVecToDict, EstimationAgentType
import LoadHealthInvData as Data

# Define a class to hold subpopulation means
class MyMeans(HARKobject):
    '''
    Class for storing subpopulation means of an outcome variable of interest.
    Stores overall mean and by income quintile, by wealth quintile, by health
    quarter, by income-wealth, and by income-health.
    '''
    def __init__(self,data,wealth,income,health):
        self.overall = np.mean(data)
        self.byIncome = np.zeros(5)
        self.byWealth = np.zeros(5)
        self.byHealth = np.zeros(4)
        self.byIncWealth = np.zeros((5,5))
        self.byIncHealth = np.zeros((5,4))
        
        for i in range(5):
            these = income == i+1
            self.byIncome[i] = np.mean(data[these])
            for j in range(5):
                those = np.logical_and(these, wealth == j+1)
                self.byIncWealth[i,j] = np.mean(data[those])
            for h in range(4):
                those = np.logical_and(these, health == h+1)
                self.byIncHealth[i,h] = np.mean(data[those])
        for j in range(5):
            these = wealth == j+1
            self.byWealth[j] = np.mean(data[these])
        for h in range(4):
            these = health == h+1
            self.byHealth[h] = np.mean(data[these])
            
    
    def subtract(self,other):
        '''
        Find the difference between the means in this instance and another MyMeans,
        returning a new instance of MyMeans
        '''
        result = deepcopy(self)
        result.overall = self.overall - other.overall
        result.byIncome = self.byIncome - other.byIncome
        result.byWealth = self.byWealth - other.byWealth
        result.byHealth = self.byHealth - other.byHealth
        result.byIncWealth = self.byIncWealth - other.byIncWealth
        result.byIncHealth = self.byIncHealth - other.byIncHealth
        return result
        
            
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
        for name in self.policy_attributes:
            for i in range(len(Agents)):
                setattr(Agents[i],name,getattr(self,name))
                
                
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
        self.t_ageInit = self.BornBoolArray.flatten() # other end of hacky fix
        self.evalExpectationFuncs()
        self.delSolution()
        
        
    def runCounterfactualAction(self):
        '''
        Run methods for extracting relevant data for a counterfactual scenario.
        '''
        self.update()
        self.solve()
        self.evalExpectationFuncs()
        self.findWTP() # NEED TO WRITE
        self.delSolution()
        
        
    def evalExpectationFuncs(self):
        '''
        Creates arrays with lifetime PDVs of several variables, from the perspective
        of the HRS subjects in 2010.  Stores arrays as attributes of self.
        '''
        # Initialize arrays, stored as attributes
        self.TotalMedPDVarray = np.nan*np.zeros(self.AgentCount)
        self.OOPmedPDVarray = np.nan*np.zeros(self.AgentCount)
        self.ExpectedLifeArray = np.nan*np.zeros(self.AgentCount)
        self.MedicarePDVarray = np.nan*np.zeros(self.AgentCount)
        self.SubsidyPDVarray = np.nan*np.zeros(self.AgentCount)
        self.WelfarePDVarray = np.nan*np.zeros(self.AgentCount)
        self.GovtPDVarray = np.nan*np.zeros(self.AgentCount)
        self.ValueArray = np.nan*np.zeros(self.AgentCount)
        
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
            self.aLvlNow[these] = self.aLvlInit[these]
            self.HlvlNow[these] = self.HlvlInit[these]
            self.getShocks()
            self.getStates()
            
            # Evaluate the PDV functions (and value function), storing in the arrays
            bLvl = self.bLvlNow[these]
            hLvl = self.hLvlNow[these]
            self.TotalMedPDVarray[these] = self.solution[t].TotalMedPDVfunc(bLvl,hLvl)
            self.OOPmedPDVarray[these] = self.solution[t].OOPmedPDVfunc(bLvl,hLvl)
            self.ExpectedLifeArray[these] = self.solution[t].ExpectedLifeFunc(bLvl,hLvl)
            self.MedicarePDVarray[these] = self.solution[t].MedicarePDVfunc(bLvl,hLvl)
            self.SubsidyPDVarray[these] = self.solution[t].SubsidyPDVfunc(bLvl,hLvl)
            self.WelfarePDVarray[these] = self.solution[t].WelfarePDVfunc(bLvl,hLvl)
            self.GovtPDVarray[these] = self.solution[t].GovtPDVfunc(bLvl,hLvl)
            self.ValueArray[these] = self.solution[t].vFunc(bLvl,hLvl)
            
            
    def findWTP(self):
        '''
        Calculate willingness-to-pay for this policy for each agent.
        '''
        pass
            
                
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
        ThisType.DataToSimRepFactor = 1
        
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
                


def calcSubpopMeans(type_list):
    '''
    Calculate overall population averages and subpopulation averages for several
    outcome variables in the Ounce of Prevention project.
    
    Parameters
    ----------
    type_list : [CounterfactualType]
        List of types used in the counterfactual experiment, which have already
        has their evalExpectationFuncs method executed.
        
    Returns
    -------
    SubpopMeans : [MyMeans]
        List of seven MyStats objects, each of which has attributes with overall
        average, average by income quintile, average by wealth quintile, average
        by health quarter, average by income-wealth, and average by income-health.
        Order: TotalMed, OOPmed, ExpectedLife, Medicare, Subsidy, Welfare, Govt.
    '''
    # Get wealth, income, and health data
    WealthQuint = np.concatenate([this_type.WealthQuint for this_type in type_list])
    IncQuint = np.concatenate([this_type.IncQuintLong for this_type in type_list])
    HealthQuarter = (np.ceil(np.concatenate([this_type.hLvlNow for this_type in type_list])*4.)).astype(int)
    
    # Get outcome data
    TotalMedPDVarray = np.concatenate([this_type.TotalMedPDVarray for this_type in type_list])
    OOPmedPDVarray = np.concatenate([this_type.OOPmedPDVarray for this_type in type_list])
    ExpectedLifeArray = np.concatenate([this_type.ExpectedLifeArray for this_type in type_list])
    MedicarePDVarray = np.concatenate([this_type.MedicarePDVarray for this_type in type_list])
    SubsidyPDVarray = np.concatenate([this_type.SubsidyPDVarray for this_type in type_list])
    WelfarePDVarray = np.concatenate([this_type.WelfarePDVarray for this_type in type_list])
    GovtPDVarray = np.concatenate([this_type.GovtPDVarray for this_type in type_list])
    
    # Make and return MyMeans objects
    TotalMedMeans = MyMeans(TotalMedPDVarray,WealthQuint,IncQuint,HealthQuarter)
    OOPmedMeans = MyMeans(OOPmedPDVarray,WealthQuint,IncQuint,HealthQuarter)
    ExpectedLifeMeans = MyMeans(ExpectedLifeArray,WealthQuint,IncQuint,HealthQuarter)
    MedicareMeans = MyMeans(MedicarePDVarray,WealthQuint,IncQuint,HealthQuarter)
    SubsidyMeans = MyMeans(SubsidyPDVarray,WealthQuint,IncQuint,HealthQuarter)
    WelfareMeans = MyMeans(WelfarePDVarray,WealthQuint,IncQuint,HealthQuarter)
    GovtMeans = MyMeans(GovtPDVarray,WealthQuint,IncQuint,HealthQuarter)
    SubpopMeans = [TotalMedMeans, OOPmedMeans, ExpectedLifeMeans, MedicareMeans, SubsidyMeans, WelfareMeans, GovtMeans]
    return SubpopMeans
    #return [TotalMedPDVarray, OOPmedPDVarray, ExpectedLifeArray, MedicarePDVarray, SubsidyPDVarray, WelfarePDVarray, GovtPDVarray]
                
                
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
    
    # Solve the baseline model and get arrays of outcome variables
    multiThreadCommands(Agents,['runBaselineAction()'],num_jobs=5)
    TotalMedBaseline, OOPmedBaseline, ExpectedLifeBaseline, MedicareBaseline, SubsidyBaseline, WelfareBaseline, GovtBaseline = calcSubpopMeans(Agents)
    for this_type in Agents:
        this_type.ValueBaseline = copy(this_type.ValueArray)
        
    #return [TotalMedBaseline, OOPmedBaseline, ExpectedLifeBaseline, MedicareBaseline, SubsidyBaseline, WelfareBaseline, GovtBaseline]
        
    # Loop through the policies, executing the counterfactuals and storing results.
    N = len(Policies)
    TotalMedDiffs = np.zeros(N)
    OOPmedDiffs = np.zeros(N)
    ExpectedLifeDiffs = np.zeros(N)
    MedicareDiffs = np.zeros(N)
    SubsidyDiffs = np.zeros(N)
    WelfareDiffs = np.zeros(N)
    GovtDiffs = np.zeros(N)
    for n in range(N):
        # Enact the policy for all of the agents
        this_policy = Policies[n]
        this_policy.enactPolicy(Agents)
        
        # Run the counterfactual and get arrays of outcome variables
        multiThreadCommands(Agents,['runCounterfactualAction()'],num_jobs=5)
        TotalMedCounterfactual, OOPmedCounterfactual, ExpectedLifeCounterfactual, MedicareCounterfactual, SubsidyCounterfactual, WelfareCounterfactual, GovtCounterfactual = calcSubpopMeans(Agents)
        
        # Calculate differences and store overall means in the arrays
        TotalMedDiff = TotalMedCounterfactual.subtract(TotalMedBaseline)
        OOPmedDiff = OOPmedCounterfactual.subtract(OOPmedBaseline)
        ExpectedLifeDiff = ExpectedLifeCounterfactual.subtract(ExpectedLifeBaseline)
        MedicareDiff = MedicareCounterfactual.subtract(MedicareBaseline)
        SubsidyDiff = SubsidyCounterfactual.subtract(SubsidyBaseline)
        WelfareDiff = WelfareCounterfactual.subtract(WelfareBaseline)
        GovtDiff = GovtCounterfactual.subtract(GovtBaseline)
        TotalMedDiffs[n] = TotalMedDiff.overall
        OOPmedDiffs[n] = OOPmedDiff.overall
        ExpectedLifeDiffs[n] = ExpectedLifeDiff.overall
        MedicareDiffs[n] = MedicareDiff.overall
        SubsidyDiffs[n] = SubsidyDiff.overall
        WelfareDiffs[n] = WelfareDiff.overall
        GovtDiffs[n] = GovtDiff.overall
        
    # If there is only one counterfactual policy, return the full set of mean-diffs.
    # If there is more than one, return vectors of overall mean-diffs.
    if len(Policies) > 1:
        return [TotalMedDiffs, OOPmedDiffs, ExpectedLifeDiffs, MedicareDiffs, SubsidyDiffs, WelfareDiffs, GovtDiffs]
    else:
        return [TotalMedDiff, OOPmedDiff, ExpectedLifeDiff, MedicareDiff, SubsidyDiff, WelfareDiff, GovtDiff]
            
            

if __name__ == '__main__':
    import HealthInvParams as Params
    from time import clock
    
    TestPolicy = SubsidyPolicy(Subsidy0=0.05,Subsidy1=0.1)
    t_start = clock()
    Out = runCounterfactuals('blah',Params.test_param_vec,[TestPolicy])
    t_end = clock()
    print('That took ' + str(t_end-t_start) + ' seconds.')
    