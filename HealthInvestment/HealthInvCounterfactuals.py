'''
This module sets up and executes counterfactual policy exercises for the Ounce of Prevention project.
'''

import sys
import os
sys.path.insert(0,'../')

from copy import copy, deepcopy
import numpy as np
from scipy.optimize import brentq
from HARKutilities import getPercentiles, plotFuncs
from HARKcore import HARKobject
from HARKparallel import multiThreadCommands, multiThreadCommandsFake
from HARKinterpolation import ConstantFunction
from HealthInvEstimation import convertVecToDict, EstimationAgentType
import LoadHealthInvData as Data
import matplotlib.pyplot as plt

# Define a class to hold subpopulation means
class MyMeans(HARKobject):
    '''
    Class for storing subpopulation means of an outcome variable of interest.
    Stores overall mean and by income quintile, by wealth quintile, by health
    quarter, by income-wealth, and by income-health.
    '''
    def __init__(self,data,wealth,income,health,sex):
        self.overall = np.mean(data)
        self.byIncome = np.zeros(5)
        self.byWealth = np.zeros(5)
        self.byHealth = np.zeros(4)
        self.byIncWealth = np.zeros((5,5))
        self.byIncHealth = np.zeros((5,4))
        self.bySex = np.zeros(2)
        self.bySexIncHealth = np.zeros((2,5,2))
        self.bySexHealth = np.zeros((2,2))
        
        half_health = np.zeros((2,health.size))
        half_health[0,:] = health <=2
        half_health[1,:] = health > 2
        
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
        for s in range(2):
            these = sex == s
            self.bySex[s] = np.mean(data[these])
            for i in range(5):
                those = np.logical_and(these, income == i+1)
                for h in range(2):
                    thise = np.logical_and(those, half_health[h,:])
                    self.bySexIncHealth[s,i,h] = np.mean(data[thise])
            for h in range(2):
                those = np.logical_and(these, half_health[h,:])
                self.bySexHealth[s,h] = np.mean(data[those])
            
    
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
    policy_attributes = ['Subsidy0','Subsidy1','FlatCopayInvst']
    
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
            if hasattr(self,name):
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
        self.evalExpectationFuncs(False)
        self.delSolution()
        
        
    def updateInsuranceFuncs(self):
        '''
        Altered version of this method that makes the counterfactuals able to
        handle a subsidy policy that specifies a flat coinsurance rate for investment.
        '''
        EstimationAgentType.updateInsuranceFuncs(self)
        if hasattr(self,'FlatCopayInvst'):
            self.CopayInvstFunc = self.T_cycle*[ConstantFunction(self.FlatCopayInvst)]
            self.SameCopayForMedAndInvst = False
        
        
    def runCounterfactualAction(self):
        '''
        Run methods for extracting relevant data for a counterfactual scenario.
        '''
        self.update()
        self.solve()
        self.evalExpectationFuncs(True)
        self.delSolution()
        
    
    def evalSocialOptimum(self):
        '''
        Execute a simulation run that calculates "socially optimal" objects.
        '''
        self.update()
        self.solve()
        self.CalcSocialOptimum = True
        self.initializeSim()
        self.simulate()
        self.iLvlSocOpt_histX = deepcopy(self.iLvlSocOpt_hist)
        self.CopaySocOpt_histX = deepcopy(self.CopaySocOpt_hist)
        self.LifePriceSocOpt_histX = deepcopy(self.LifePriceSocOpt_hist)
        self.CumLivPrb_histX = deepcopy(self.CumLivPrb_hist)
        self.CalcSocialOptimum = False
        
        OptimalCopayInvstFunc = []
        OptCopayList = []
        for t in range(self.T_cycle):
            Weights = self.CumLivPrb_hist[t,:]
            Copays  = self.CopaySocOpt_hist[t,:]
            these = np.logical_not(np.isnan(Copays))
            OptCopay = np.dot(Weights[these],Copays[these])/np.sum(Weights[these])
            if np.isnan(OptCopay):
                OptCopay = 1.0
            OptimalCopayInvstFunc.append(ConstantFunction(OptCopay))
            OptCopayList.append(OptCopay)
        self.OptimalCopayInvstFunc = OptimalCopayInvstFunc
        self.OptCopayList = OptCopayList
        
        self.delSolution()
        
        
    def evalExpectationFuncs(self,isCounterfactual):
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
        self.WTParray = np.nan*np.zeros(self.AgentCount)
        
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
            if isCounterfactual:
                self.WTParray[these] = self.findWTP(t,these)
            
            
    def findWTP(self,t,these):
        '''
        Calculate willingness-to-pay for this policy for each agent selected.
        '''
        # Lazy loop to calculate WTP; can be sped up later
        WTParray = np.zeros(np.sum(these))
        j = 0
        for i in range(self.AgentCount):
            if these[i]:
                v = self.ValueBaseline[i]
                b = self.bLvlNow[i]
                h = self.hLvlNow[i]
                f = lambda x : self.solution[t].vFunc(b - x, h*np.ones_like(x)) - v
                X = np.linspace(0.,b,20)
                fX = f(X)
                real_test = np.logical_and(np.isreal(fX), np.logical_not(np.isnan(fX)))
                bot = -b # This should never happen, means someone is made $10k worse off by subsidy
                top = np.max(X[real_test])
                #print(i,b,h,top,f(bot),f(top))
                try:
                    WTParray[j] = brentq(f,bot,top)
                except:
                    WTParray[j] = 0.0
                j += 1
        return WTParray
            
                
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
        ThisType.DataToSimRepFactor = 50
        
        these = Data.TypeBoolArrayCounterfactual[n,:]
        ThisType.DataAgentCount = np.sum(these)
        ThisType.WealthQuint = Data.wealth_quint_data[these]
        ThisType.HealthTert = Data.health_tert_data[these]
        ThisType.HealthQuint = Data.health_quint_data[these]
        ThisType.aLvlInit = Data.w7_data[these]
        ThisType.HlvlInit = Data.h7_data[these]
        ThisType.t_ageInit = Data.age_in_2010[these]
        ThisType.BornBoolArray = Data.BornBoolArray[:,these]
        ThisType.InDataSpanArray = np.zeros(10) # irrelevant
        ThisType.CalcExpectationFuncs = True
        ThisType.track_vars = ['OOPmedNow','hLvlNow','aLvlNow','CumLivPrb','DiePrbNow',
                               'RatioNow','MedLvlNow','CopayMedNow','CopayInvstNow',
                               'iLvlNow','iLvlSocOpt','CopaySocOpt','LifePriceSocOpt']
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
    Sex = np.concatenate([this_type.SexLong for this_type in type_list])
    HealthQuarter = (np.ceil(np.concatenate([this_type.hLvlNow for this_type in type_list])*4.)).astype(int)
    
    # Get outcome data
    TotalMedPDVarray = np.concatenate([this_type.TotalMedPDVarray for this_type in type_list])
    OOPmedPDVarray = np.concatenate([this_type.OOPmedPDVarray for this_type in type_list])
    ExpectedLifeArray = np.concatenate([this_type.ExpectedLifeArray for this_type in type_list])
    MedicarePDVarray = np.concatenate([this_type.MedicarePDVarray for this_type in type_list])
    SubsidyPDVarray = np.concatenate([this_type.SubsidyPDVarray for this_type in type_list])
    WelfarePDVarray = np.concatenate([this_type.WelfarePDVarray for this_type in type_list])
    GovtPDVarray = np.concatenate([this_type.GovtPDVarray for this_type in type_list])
    WTParray = np.concatenate([this_type.WTParray for this_type in type_list])
    
    # Make and return MyMeans objects
    TotalMedMeans = MyMeans(TotalMedPDVarray,WealthQuint,IncQuint,HealthQuarter,Sex)
    OOPmedMeans = MyMeans(OOPmedPDVarray,WealthQuint,IncQuint,HealthQuarter,Sex)
    ExpectedLifeMeans = MyMeans(ExpectedLifeArray,WealthQuint,IncQuint,HealthQuarter,Sex)
    MedicareMeans = MyMeans(MedicarePDVarray,WealthQuint,IncQuint,HealthQuarter,Sex)
    SubsidyMeans = MyMeans(SubsidyPDVarray,WealthQuint,IncQuint,HealthQuarter,Sex)
    WelfareMeans = MyMeans(WelfarePDVarray,WealthQuint,IncQuint,HealthQuarter,Sex)
    GovtMeans = MyMeans(GovtPDVarray,WealthQuint,IncQuint,HealthQuarter,Sex)
    WTPmeans = MyMeans(WTParray,WealthQuint,IncQuint,HealthQuarter,Sex)
    SubpopMeans = [TotalMedMeans, OOPmedMeans, ExpectedLifeMeans, MedicareMeans, SubsidyMeans, WelfareMeans, GovtMeans, WTPmeans]
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
    TotalMedBaseline, OOPmedBaseline, ExpectedLifeBaseline, MedicareBaseline, SubsidyBaseline, WelfareBaseline, GovtBaseline, trash = calcSubpopMeans(Agents)
    for this_type in Agents:
        this_type.ValueBaseline = copy(this_type.ValueArray)
    print('Finished the baseline policy.')
           
    # Loop through the policies, executing the counterfactuals and storing results.
    N = len(Policies)
    TotalMedDiffs = np.zeros(N)
    OOPmedDiffs = np.zeros(N)
    LifeDiffs = np.zeros(N)
    MedicareDiffs = np.zeros(N)
    SubsidyDiffs = np.zeros(N)
    WelfareDiffs = np.zeros(N)
    GovtDiffs = np.zeros(N)
    WTPs = np.zeros(N)
    LifeDiffsByIncome = np.zeros((N,5))
    WTPsByIncome = np.zeros((N,5))
    
    for n in range(N):
        # Enact the policy for all of the agents
        this_policy = Policies[n]
        this_policy.enactPolicy(Agents)
        
        # Run the counterfactual and get arrays of outcome variables
        multiThreadCommands(Agents,['runCounterfactualAction()'],num_jobs=5)
        TotalMedCounterfactual, OOPmedCounterfactual, ExpectedLifeCounterfactual, MedicareCounterfactual, SubsidyCounterfactual, WelfareCounterfactual, GovtCounterfactual, WTPcounterfactual = calcSubpopMeans(Agents)
        
        # Calculate differences and store overall means in the arrays
        TotalMedDiff = TotalMedCounterfactual.subtract(TotalMedBaseline)
        OOPmedDiff = OOPmedCounterfactual.subtract(OOPmedBaseline)
        LifeDiff = ExpectedLifeCounterfactual.subtract(ExpectedLifeBaseline)
        MedicareDiff = MedicareCounterfactual.subtract(MedicareBaseline)
        SubsidyDiff = SubsidyCounterfactual.subtract(SubsidyBaseline)
        WelfareDiff = WelfareCounterfactual.subtract(WelfareBaseline)
        GovtDiff = GovtCounterfactual.subtract(GovtBaseline)
        TotalMedDiffs[n] = TotalMedDiff.overall
        OOPmedDiffs[n] = OOPmedDiff.overall
        LifeDiffs[n] = LifeDiff.overall
        for i in range(5):
            LifeDiffsByIncome[n,i] = LifeDiff.byIncome[i]
        MedicareDiffs[n] = MedicareDiff.overall
        SubsidyDiffs[n] = SubsidyDiff.overall
        WelfareDiffs[n] = WelfareDiff.overall
        GovtDiffs[n] = GovtDiff.overall
        WTPs[n] = WTPcounterfactual.overall
        for i in range(5):
            WTPsByIncome[n,i] = WTPcounterfactual.byIncome[i]
        print('Finished counterfactual policy ' + str(n+1) + ' of ' + str(N) + ' for ' + name +  '.')
        
    # If there is only one counterfactual policy, return the full set of mean-diffs.
    # If there is more than one, return vectors of overall mean-diffs.
    if len(Policies) > 1:
        return [TotalMedDiffs, OOPmedDiffs, LifeDiffs, MedicareDiffs, SubsidyDiffs, WelfareDiffs, GovtDiffs, WTPs, LifeDiffsByIncome, WTPsByIncome]
    else:
        return [TotalMedDiff, OOPmedDiff, LifeDiff, MedicareDiff, SubsidyDiff, WelfareDiff, GovtDiff, WTPcounterfactual, ExpectedLifeBaseline]
    
    
# Define a function for evaluating the "socially optimal" policy
def runOptimalPolicy(Parameters, LifePrice):
    '''
    Solve for the "socially optimal" health investment subsidy policy, then
    simulate its effects; depends on user-specified value of a year of life.
    
    Parameters
    ----------
    name : str
        Name of this counterfactual set, used in filenames.
    Parameters : np.array
        A size 33 array of parameters, just like for the estimation.
    LifePrice : float
        Exogenous dollar "value" of a year of life, in units of $10,000.
    
    Returns
    -------
    TBD
    '''
    # Make the agent types
    param_dict = convertVecToDict(Parameters)
    Agents = makeMultiTypeCounterfactual(param_dict)
    for this_type in Agents:
        this_type.LifePrice = copy(LifePrice)
    
    # Solve the baseline model and get arrays of outcome variables
    multiThreadCommands(Agents,['runBaselineAction()'],num_jobs=5)
    TotalMedBaseline, OOPmedBaseline, LifeBaseline, MedicareBaseline, SubsidyBaseline, WelfareBaseline, GovtBaseline, trash = calcSubpopMeans(Agents)
    for this_type in Agents:
        this_type.ValueBaseline = copy(this_type.ValueArray)
        
    # Find the "socially optimal" policy and implement it
    multiThreadCommands(Agents,['evalSocialOptimum()'],num_jobs=5)
    for this_type in Agents:
        this_type.CopayInvstFunc = this_type.OptimalCopayInvstFunc
        this_type.SameCopayForMedAndInvst = False # Make sure CopayInvst isn't overwritten!
        
    # Run the counterfactual and get arrays of outcome variables
    multiThreadCommands(Agents,['runCounterfactualAction()'],num_jobs=5)
    TotalMedCounterfactual, OOPmedCounterfactual, LifeCounterfactual, MedicareCounterfactual, SubsidyCounterfactual, WelfareCounterfactual, GovtCounterfactual, WTPcounterfactual = calcSubpopMeans(Agents)
    
    # Calculate differences and store overall means in the arrays
    TotalMedDiff = TotalMedCounterfactual.subtract(TotalMedBaseline)
    OOPmedDiff = OOPmedCounterfactual.subtract(OOPmedBaseline)
    LifeDiff = LifeCounterfactual.subtract(LifeBaseline)
    MedicareDiff = MedicareCounterfactual.subtract(MedicareBaseline)
    SubsidyDiff = SubsidyCounterfactual.subtract(SubsidyBaseline)
    WelfareDiff = WelfareCounterfactual.subtract(WelfareBaseline)
    GovtDiff = GovtCounterfactual.subtract(GovtBaseline)
        
    # Return the full set of mean-diffs.
    return [TotalMedDiff, OOPmedDiff, LifeDiff, MedicareDiff, SubsidyDiff, WelfareDiff, GovtDiff, WTPcounterfactual]


# Define a function for evaluating the "socially optimal" policy
def runOptimalPolicies(name, Parameters, LifePriceVec):
    '''
    Solve for the "socially optimal" health investment subsidy policy, then
    simulate its effects; depends on user-specified value of a year of life.
    
    Parameters
    ----------
    name : str
        Name of this counterfactual set, used in filenames.
    Parameters : np.array
        A size 33 array of parameters, just like for the estimation.
    LifePriceVec : np.array
        1D array of exogenous dollar "value" of a year of life, in units of $10,000.
    
    Returns
    -------
    TBD
    '''
    # Initialize the output arrays
    N = LifePriceVec.size
    TotalMedDiffs = np.zeros(N)
    OOPmedDiffs = np.zeros(N)
    LifeDiffs = np.zeros(N)
    MedicareDiffs = np.zeros(N)
    SubsidyDiffs = np.zeros(N)
    WelfareDiffs = np.zeros(N)
    GovtDiffs = np.zeros(N)
    WTPs = np.zeros(N)
    LifeDiffsByIncome = np.zeros((N,5))
    WTPsByIncome = np.zeros((N,5))
    
    # Loop through the values of LifePriceVec and fill in the output
    for n in range(N):
        LifePrice = LifePriceVec[n]
        TotalMedDiff, OOPmedDiff, LifeDiff, MedicareDiff, SubsidyDiff, WelfareDiff, GovtDiff, WTPcounterfactual = runOptimalPolicy(Parameters, LifePrice)
        
        TotalMedDiffs[n] = TotalMedDiff.overall
        OOPmedDiffs[n] = OOPmedDiff.overall
        LifeDiffs[n] = LifeDiff.overall
        for i in range(5):
            LifeDiffsByIncome[n,i] = LifeDiff.byIncome[i]
        MedicareDiffs[n] = MedicareDiff.overall
        SubsidyDiffs[n] = SubsidyDiff.overall
        WelfareDiffs[n] = WelfareDiff.overall
        GovtDiffs[n] = GovtDiff.overall
        WTPs[n] = WTPcounterfactual.overall
        for i in range(5):
            WTPsByIncome[n,i] = WTPcounterfactual.byIncome[i]
        print('Finished counterfactual policy ' + str(n+1) + ' of ' + str(N) + ' for ' + name +  '.')
        
    return [TotalMedDiffs, OOPmedDiffs, LifeDiffs, MedicareDiffs, SubsidyDiffs, WelfareDiffs, GovtDiffs, WTPs, LifeDiffsByIncome, WTPsByIncome]


if __name__ == '__main__':
    import HealthInvParams as Params
    from time import clock
    from MakeTables import makeCounterfactualSummaryTables
    from MakeFigures import makeCounterfactualFigures
    
    # Choose which policy experiments to run
    run_test_policy = False
    run_universal    = False
    run_preventive  = False
    run_curative    = False
    run_flat_copay  = False
    run_optimal     = False
    
    
    if run_test_policy:
        # Run a test policy experiment
        TestPolicy = SubsidyPolicy(Subsidy0=10*[0.1],Subsidy1=10*[0.0])
        t_start = clock()
        Out = runCounterfactuals('blah',Params.test_param_vec,[TestPolicy])
        t_end = clock()
        print('The text policy experiment took ' + str(t_end-t_start) + ' seconds.')
        makeCounterfactualSummaryTables(Out,'Test Policy','testname','Test')

    if run_universal:
        # Run an experiment in which a direct universal subsidy is used
        PolicyList = []
        SubsidyVec = np.linspace(0,0.6,51)
        for x in SubsidyVec:
            PolicyList.append(SubsidyPolicy(Subsidy0=10*[x],Subsidy1=10*[0.0]))
        t_start = clock()
        Out = runCounterfactuals('universal subsidy experiment',Params.test_param_vec,PolicyList)
        t_end = clock()
        print('The universal subsidy experiment took ' + str(t_end-t_start) + ' seconds.')
        makeCounterfactualFigures(Out,SubsidyVec,'Universal subsidy (y2000 USD)', 'universal subsidy', 'UniversalSub')
       
    if run_preventive:
        # Run an experiment in which only preventive care is subsidized
        PolicyList = []
        SubsidyVec = np.linspace(0,0.6,51)
        for x in SubsidyVec:
            PolicyList.append(SubsidyPolicy(Subsidy0=10*[x], SubsidyHealthCutoff=10*[0.5], PreventiveSubsidy=10*[True]))
        t_start = clock()
        Out = runCounterfactuals('preventive care subsidy experiment',Params.test_param_vec,PolicyList)
        t_end = clock()
        print('The preventive care subsidy experiment took ' + str(t_end-t_start) + ' seconds.')
        makeCounterfactualFigures(Out,SubsidyVec,'Preventive care subsidy (y2000 USD)', 'preventive care subsidy', 'PreventiveSub')

    if run_curative:
        # Run an experiment in which only curative care is subsidized
        PolicyList = []
        SubsidyVec = np.linspace(0,0.6,51)
        for x in SubsidyVec:
            PolicyList.append(SubsidyPolicy(Subsidy0=10*[x], SubsidyHealthCutoff=10*[0.5], PreventiveSubsidy=10*[False]))
        t_start = clock()
        Out = runCounterfactuals('curative care subsidy experiment',Params.test_param_vec,PolicyList)
        t_end = clock()
        print('The curative care subsidy experiment took ' + str(t_end-t_start) + ' seconds.')
        makeCounterfactualFigures(Out,SubsidyVec,'Curative care subsidy (y2000 USD)', 'curative care subsidy', 'CurativeSub')
        
    if run_flat_copay:
        # Run an experiment in which the coinsurance rate on health investment is varied
        PolicyList = []
        CopayVec = np.linspace(0.02,1.0,51)
        for x in CopayVec:
            PolicyList.append(SubsidyPolicy(FlatCopayInvst=10*[x]))
        t_start = clock()
        Out = runCounterfactuals('flat coinsurance rate experiment',Params.test_param_vec,PolicyList)
        t_end = clock()
        print('The flat coinsurance rate experiment took ' + str(t_end-t_start) + ' seconds.')
        makeCounterfactualFigures(Out,CopayVec, r'Coinsurance rate for health investment $q^{n}$', 'flat copay', 'FlatCopayInvst')
        
    if run_optimal:    
        # Run an experiment using the "optimal" investment policy NEED TO DO
        t_start = clock()
        LifePriceVec = np.linspace(0.,20.,51)
        Out = runOptimalPolicies('socially optimal policy experiment', Params.test_param_vec, LifePriceVec)
        t_end = clock()
        print('The "socially optimal policy" experiment took ' + str(t_end-t_start) + ' seconds.')
        makeCounterfactualFigures(Out, LifePriceVec, r'Value of a year of life $P_L$, \$10,000 (y2000)', 'socially optimal', 'SocOptByLifePrice')
        
        
    # TODO BY END OF JULY 15:
    # XX Set up experiment for subsidy on curative care
    # XX Set up experiment for subsidy on preventive care
    # XX Write function to run experiment where LifePrice varies --> optimal policy
    # 4) Check optimal insurance policy procedure THURS
    # 5) Make figures of PDV of TotalMed by health (across income and sex) WEDS
    # XX Write "dummy" solver that generates trivial policy functions
    # 7) Re-estimate health and mortality moments with ^^ and no investment FRI
    # 8) Make moment fit figure for health moments for that FRI
    # XX Make alternate version of moment category 7(a) moment figure
    # 10) Write health investment fixed point appendix WEDS
    # 11) Write liquidity constrained solution appendix FRI
    # 12) Write consumption floor solution appendix FRI
    # 13) Write G2EGM appendix SATURDAY
    # 14) Write integration method / accounting appendix RETURN FLIGHT
    # 15) Write short paragraph for moment fit appendix WEDS
    # XX  Make counterfactual figure template in paper
    # 17) Write identification arguments for HealthProdParam, etc SUNDAY
    # 18) Make counterfactual tables FRIDAY
    # 19) Make health ordered probit and insurance function tables FRIDAY
    # 20) Move code into paper archive
    
    # BWI --> AUS (1), (2), (3), (6), (9), (16)
    # July 11: (5), (10), (15)
    # July 12: (4)
    # AUS --> BWI: (14)
    # July 13: (7), (8), (11), (12), (18), (19)
    # July 14: (13), begin (17)
    # July 15: finish (17), (20)
    
    # -----------
    
    # July 16: Write counterfactual procedure and outcomes of interest section
    # July 17: Write up results for various subsidy policies
    # July 18: Write discussion of "socially optimal policy"
    # July 19: Write discussion of PDV functions and derivative wrt health
    # July 20: Address small referee points, Section 5 mop up
    
    # July 21-22: Move houses
    
    # -----------
    
    # July 23: Red pen introduction and literature, incorporate small referee suggestions in lit
    # July 24-25: Rework / reframe introduction
    # July 26: Rewrite abstract, conclusion, check internal and external references
    # July 27: Full proofread, begin notebook
    # July 28-29: Get notebook functioning on mybinder, begin letter to Matthias
    # July 30-31: Finalize letter and submit

