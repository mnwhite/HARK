'''
This module runs counterfactual experiments on the structure of insurance market rules.
'''

import sys 
sys.path.insert(0,'../../')

import numpy as np
import csv
from copy import copy, deepcopy
from DynInsSelEstimation import makeMarketFromParams
from ActuarialRules import flatActuarialRule, exclusionaryActuarialRule, healthRatedActuarialRule, ageHealthRatedActuarialRule, ageRatedActuarialRule
from HARKutilities import getPercentiles, kernelRegression
from HARKparallel import multiThreadCommands, multiThreadCommandsFake
import matplotlib.pyplot as plt

def runCounterfactual(Parameters,Baseline,Counterfactuals):
    '''
    Runs a counterfactual experiment comparing individuals' welfare under two
    actuarial rules.  Constructs a market for the agents to live in, and solves
    it under the "before" rule, generating distribution of individual states.
    It then evaluates the value function for each of the simulated agents, saving
    it as an array.  It then solves the market under the "after" rule, keeping
    the value function(s) again.  For each simulated agent state from the "before"
    world, it does a search for the pLvl_comp where the agent is indifferent be-
    tween the rules: vAfter(mLvl,pLvl_comp,h) = vBefore(mLvl,pLvl,h).  Returns the
    market itself as output, and saves counterfactual data to a CSV file.
    
    Parameters
    ----------
    Parameters : np.array
        Array of 33 structural parameters, just like objectiveFunction.
    Baseline : ActuarialSpecification
        Specification of the actuarial rule for the "before" state.
    Counterfactuals : [ActuarialSpecification]
        Specification(s) of the actuarial rule for the counterfactual world.
        
    Returns
    -------
    MyMarket : DynInsSelMarket
        Market that was used to run the counterfactual experiment.
    mLvlArray : np.array
        2D array of market resource levels at different ages.
    pLvlArray : np.array
        2D array of permanent income levels at different ages.
    hArray : np.array
        2D integer array of discrete health states at different ages.
    CompVarArray : np.array
        2D array of compensating variation as a fraction of permanent income.
    '''
    FigsDir = '../CounterfactualFigs/'
    
    EvalType = 3 # Number of times to do a static search for eqbm premiums
    InsChoice = 1 # Extent of insurance choice
    SubsidyTypeCount = 0 # Number of discrete non-zero subsidy levels
    CRRAtypeCount = 1 # Number of CRRA types (DON'T USE)
    ZeroSubsidyBool = True # Whether to include a zero subsidy type
    
    OrigPremiums = np.array([0.5200, 0.0, 0.0, 0.0, 0.0])
    
    # Make a basic array of (arbitrary) initial premiums
    ContractCounts = [0,1,5] # plus one
    Premiums_init_short = np.concatenate((np.array([0.]),OrigPremiums[0:ContractCounts[InsChoice]]))
    Premiums_init = np.tile(np.reshape(Premiums_init_short,(1,Premiums_init_short.size)),(40,1))
    Premiums_init = np.vstack((Premiums_init,np.zeros((20,ContractCounts[InsChoice]+1))))
    
    # Make the market, giving it the "before" rule
    MyMarket = makeMarketFromParams(Parameters,Baseline.ActuarialRule,Premiums_init,InsChoice,SubsidyTypeCount,CRRAtypeCount,ZeroSubsidyBool)
    MyMarket.HealthStateGroups = Baseline.HealthStateGroups
    MyMarket.AgeBandLimit = Baseline.AgeBandLimit
    for this_type in MyMarket.agents:
        this_type.updateUninsuredPremium(Baseline.MandateTax)
    MyMarket.Premiums = Premiums_init_short
    MyMarket.LoadFac = 1.5
    multiThreadCommands(MyMarket.agents,['update()','makeShockHistory()'])
    MyMarket.getIncomeQuintiles()
    multiThreadCommandsFake(MyMarket.agents,['makeIncBoolArray()'])
    
    # Solve the market with the "before" rule
    MyMarket.max_loops = EvalType
    MyMarket.solve()

    # Extract history arrays of m,p,h,v for all agents in the "before" market
    multiThreadCommandsFake(MyMarket.agents,['makevHistArray()'])
    vHistBefore = np.concatenate([this_type.vNow_hist for this_type in MyMarket.agents],axis=1)
    mHistBefore = np.concatenate([this_type.mLvlNow_hist for this_type in MyMarket.agents],axis=1)
    pHist = np.concatenate([this_type.pLvlHist for this_type in MyMarket.agents],axis=1)
    hBoolArray = np.concatenate([this_type.HealthBoolArray for this_type in MyMarket.agents],axis=1)
    hHist = -np.ones_like(vHistBefore,dtype=int)
    iHistBefore = np.concatenate([this_type.ContractNow_hist for this_type in MyMarket.agents],axis=1) > 0
    for h in range(hBoolArray.shape[2]):
        these = hBoolArray[:,:,h]
        hHist[these] = h
    T = MyMarket.agents[0].T_sim
    N = mHistBefore.shape[1]
    ageHist = np.tile(np.reshape(np.arange(T),(T,1)),(1,N))
    
    # Make a plot of premiums by age and health state
    PremiumsBefore = MyMarket.Premiums
    if len(PremiumsBefore.shape) == 1:
        plt.plot([25,64],[10*PremiumsBefore[1],10*PremiumsBefore[1]],'-b')
    elif len(PremiumsBefore.shape) == 2:
        plt.plot(np.arange(25,65),10*PremiumsBefore[1,:],'-b')
    elif len(PremiumsBefore.shape) == 3:
        for g in range(0,PremiumsBefore.shape[1]):
            plt.plot(np.arange(25,65),10*PremiumsBefore[1,g,:],'-')
    plt.xlabel('Age')
    plt.ylabel('Annual premium (thousands of USD)')
    plt.title('Premiums by age in baseline scenario')
    plt.savefig(FigsDir + 'PremiumsBaseline.pdf')
    plt.show()
    
    for Counterfactual in Counterfactuals:
        # Now give the market the "after" rule, and have it update premiums using the old distribution, then solve it properly
        MyMarket.ActuarialRule = Counterfactual.ActuarialRule
        MyMarket.HealthStateGroups = Counterfactual.HealthStateGroups
        MyMarket.AgeBandLimit = Counterfactual.AgeBandLimit
        for this_type in MyMarket.agents: # Give agents the counterfactual insurance mandate
            this_type.updateUninsuredPremium(Counterfactual.MandateTax)
        MyMarket.makeHistory()
        MyMarket.updateDynamics() # make sure agents actually get the new premiums
        MyMarket.solve()
        
        # Make a plot of premiums by age and health state
        PremiumsAfter = MyMarket.Premiums
        if len(PremiumsAfter.shape) == 1:
            plt.plot([25,64],[10*PremiumsAfter[1],10*PremiumsAfter[1]],'-b')
        elif len(PremiumsAfter.shape) == 2:
            plt.plot(np.arange(25,65),10*PremiumsAfter[1,:],'-b')
        elif len(PremiumsAfter.shape) == 3:
            for g in range(0,PremiumsAfter.shape[1]):
                plt.plot(np.arange(25,65),10*PremiumsAfter[1,g,:],'-')
        plt.xlabel('Age')
        plt.ylabel('Annual premium (thousands of USD)')
        plt.title('Premiums by age, ' + Counterfactual.text)
        plt.ylim(0.,11.)
        plt.savefig(FigsDir + 'Premiums' + Counterfactual.name + '.pdf')
        plt.show()
        
        # Replace the simulated histories of the after market with those of the before market
        K = 0 # Running count of position in big arrays
        for this_type in MyMarket.agents:
            this_type.mLvlNow_hist = mHistBefore[:,K:(K+this_type.AgentCount)]
            this_type.vTarg_hist = vHistBefore[:,K:(K+this_type.AgentCount)]
            K += this_type.AgentCount
            # Each type should have the same pLvlHist and HealthBoolArray before and after (don't need to change it)
            
        # Find compensating variation for each simulated individual, in terms of pLvl (and extract the results)
        multiThreadCommands(MyMarket.agents,['findCompensatingpLvl()'])
        pComp_all = np.concatenate([this_type.pCompHist for this_type in MyMarket.agents],axis=1)
        Invalid_all = np.concatenate([this_type.pComp_invalid for this_type in MyMarket.agents],axis=1)
        iHistAfter = np.concatenate([this_type.ContractNow_hist for this_type in MyMarket.agents],axis=1) > 0
        
        # Reshape the useable data into 1D arrays and combine
        which = np.logical_and(hHist >= 0,ageHist < 40)
        mLvl = mHistBefore[which]
        pLvl = pHist[which]
        health = hHist[which]
        age = ageHist[which]
        pComp = pComp_all[which]
        Invalid = Invalid_all[which]
        iBefore = iHistBefore[which]
        iAfter = iHistAfter[which]
        counterfactual_data = np.vstack([age,health,mLvl,pLvl,pComp,Invalid,iBefore,iAfter]).transpose()
        
        # Write the counterfactual results to a CSV file
        VarNames = ['age','health','mLvl','pLvl','pComp','Invalid','iBefore','iAfter']
        with open('../Results/' + Counterfactual.name + 'Data.txt.','wb') as f:
            my_writer = csv.writer(f, delimiter = '\t')
            my_writer.writerow(VarNames)
            for i in range(counterfactual_data.shape[0]):
                X = counterfactual_data[i,:]
                this_row = ['%.0f' % X[0], '%.0f' % X[1], '%.4f' % X[2], '%.4f' % X[3], '%.4f' % X[4], '%.0f' % X[5], '%.0f' % X[6], '%.0f' % X[7]]
                my_writer.writerow(this_row)
            f.close()
            
    return MyMarket


def makeCounterfactualFigures(specification):
    '''
    Produces many figures to graphically represent the results of a counterfactual
    experiment and saves them to the folder ../CounterfactualFigures.
    
    Parameters
    ----------
    specification : ActuarialSpecification
        Counterfactual specification whose figures are to be produced.
        
    Returns
    -------
    None
    '''
    with open('../Results/' + specification.name + 'Data.txt','r') as f:
        my_reader = csv.reader(f, delimiter = '\t')
        all_data = list(my_reader)
    FigsDir = '../CounterfactualFigs/'
        
    N = len(all_data) - 1
    T = 40
    age = np.zeros(N,dtype=int)
    health = np.zeros(N,dtype=int)
    mLvl = np.zeros(N,dtype=float)
    pLvl = np.zeros(N,dtype=float)
    pComp = np.zeros(N,dtype=float)
    invalid = np.zeros(N,dtype=bool)
    iBefore = np.zeros(N,dtype=bool)
    iAfter = np.zeros(N,dtype=bool)
    
    for i in range(N):
        j = i+1
        age[i] = int(float(all_data[j][0]))
        health[i] = int(float(all_data[j][1]))
        mLvl[i] = float(all_data[j][2])
        pLvl[i] = float(all_data[j][3])
        pComp[i] = float(all_data[j][4])
        invalid[i] = bool(float(all_data[j][5]))
        iBefore[i] = bool(float(all_data[j][6]))
        iAfter[i] = bool(float(all_data[j][7]))  
    WTP = 1. - pComp/pLvl
    valid = np.logical_not(invalid)
        
    # Make a quantile plot of permanent income by age
    pctiles = [0.01,0.10,0.25,0.50,0.75,0.90,0.99]
    AgeVec = np.arange(T,dtype=int) + 25
    pLvlQuantilesByAge = np.zeros((T,len(pctiles)))
    for t in range(T):
        these = age == t
        pLvlQuantilesByAge[t,:] = getPercentiles(pLvl[these],percentiles=pctiles)
    plt.plot(AgeVec,pLvlQuantilesByAge*10,'-k')
    plt.xlabel('Age')
    plt.ylabel('Permanent income level (thousands of USD)')
    plt.title('Quantiles of permanent income by age')
    plt.savefig(FigsDir + 'PermIncQuantiles.pdf')
    plt.show()
    
    # Make a plot of insured rate by age (and health) in the before scenario
    InsuredRateByAge = np.zeros(T)
    InsuredRateByAgeHealth = np.zeros((T,5))
    for t in range(T):
        these = age == t
        InsuredRateByAge[t] = float(np.sum(iBefore[these]))/float(np.sum(these))
        for h in range(5):
            those = np.logical_and(these, health == h)
            InsuredRateByAgeHealth[t,h] = float(np.sum(iBefore[those]))/float(np.sum(those))
    plt.plot(AgeVec,InsuredRateByAge,'--k')
    plt.plot(AgeVec,InsuredRateByAgeHealth,'-')
    plt.xlabel('Age')
    plt.ylabel('Insured rate by health status')
    plt.title('Insured rate, baseline scenario')
    plt.ylim(0.,1.)
    plt.savefig(FigsDir + 'InsuredRateBaseline.pdf')
    plt.show()
    
    # Make a plot of insured rate by age (and health) in the after scenario
    InsuredRateByAge = np.zeros(T)
    InsuredRateByAgeHealth = np.zeros((T,5))
    for t in range(T):
        these = age == t
        InsuredRateByAge[t] = float(np.sum(iAfter[these]))/float(np.sum(these))
        for h in range(5):
            those = np.logical_and(these, health == h)
            InsuredRateByAgeHealth[t,h] = float(np.sum(iAfter[those]))/float(np.sum(those))
    plt.plot(AgeVec,InsuredRateByAge,'--k')
    plt.plot(AgeVec,InsuredRateByAgeHealth,'-')
    plt.xlabel('Age')
    plt.ylabel('Insured rate by health status')
    plt.title('Insured rate, ' + specification.text)
    plt.ylim(0.,1.)
    plt.savefig(FigsDir + 'InsuredRate' + specification.name + '.pdf')
    plt.show()
        
    # Make a quantile plot of willingness to pay by age
    WTPquantilesByAge = np.zeros((T,len(pctiles)))
    for t in range(T):
        these = np.logical_and(age == t, valid)
        WTPquantilesByAge[t,:] = getPercentiles(WTP[these],percentiles=pctiles)
    plt.plot(AgeVec,WTPquantilesByAge*100,'-k')
    plt.xlabel('Age')
    plt.ylabel('Willingness-to-pay (% of permanent income)')
    plt.title('Quantiles of willingness-to-pay by age, ' + specification.text)
    plt.savefig(FigsDir + 'WTPquantiles' + specification.name + '.pdf')
    plt.show()
    
    # Make a plot of mean willingness to pay by age and health
    WTPmeanByAgeHealth = np.zeros((T,5)) + np.nan
    for t in range(T):
        these = np.logical_and(age == t, valid)
        for h in range(5):
            those = np.logical_and(these,health==h)
            WTPmeanByAgeHealth[t,h] = np.mean(WTP[those])
    plt.plot(AgeVec,WTPmeanByAgeHealth*100,'-')
    plt.xlabel('Age')
    plt.ylabel('Willingness-to-pay (% of permanent income)')
    plt.title('Mean willingness-to-pay by age and health, ' + specification.text)
    plt.savefig(FigsDir + 'WTPbyAgeHealth' + specification.name + '.pdf')
    plt.show()
    
    # Make a plot of kernel regression of willingness to pay by permanent income and age (in 5 year age blocks)
    for a in range(8):
        age_min = a*5
        age_max = age_min+5
        these = np.logical_and(np.logical_and(age >= age_min,age < age_max), valid)
        p_temp = pLvl[these]
        cuts = getPercentiles(p_temp,percentiles=[0.01,0.99])
        f = kernelRegression(p_temp,WTP[these],bot=cuts[0],top=20.,N=200,h=0.5)
        P = np.linspace(cuts[0],20.,200)
        plt.plot(P*10,f(P)*100,'-')
    plt.xlabel('Permanent income level (thousands of USD)')
    plt.ylabel('Willingness-to-pay (% of permanent income)')
    plt.title('Mean willingness-to-pay by income and age, ' + specification.text)
    plt.savefig(FigsDir + 'WTPbyAgeIncome' + specification.name + '.pdf')
    plt.show()
    
    # Make a plot of kernel regression of willingness to pay by permanent income and health
    for h in range(5):
        these = np.logical_and(health==h, valid)
        p_temp = pLvl[these]
        cuts = getPercentiles(p_temp,percentiles=[0.01,0.99])
        f = kernelRegression(p_temp,WTP[these],bot=cuts[0],top=20.,N=200,h=0.5)
        P = np.linspace(cuts[0],20.,200)
        plt.plot(P*10,f(P)*100,'-')
    plt.xlabel('Permanent income level (thousands of USD)')
    plt.ylabel('Willingness-to-pay (% of permanent income)')
    plt.title('Mean willingness-to-pay by income and health, ' + specification.text)
    plt.savefig(FigsDir + 'WTPbyIncomeHealth' + specification.name + '.pdf')
    plt.show()
        

class ActuarialSpecification(object):
    '''
    A very simple class for representing specifications for counterfactuals.
    Attributes include an ActuarialRule function, HealthStateGroups, AgeBandLimit,
    a MandateTax, and a name.  Each of these attributes is passed to the Market.
    '''
    def __init__(self,ActuarialRule,HealthStateGroups,AgeBandLimit,MandateTax,name,text):
        self.ActuarialRule = ActuarialRule
        self.HealthStateGroups = HealthStateGroups
        self.AgeBandLimit = AgeBandLimit
        self.MandateTax = MandateTax
        self.name = name
        self.text = text


# Define the baseline specification
BaselineSpec = ActuarialSpecification(
                        ActuarialRule = ageHealthRatedActuarialRule,
                        HealthStateGroups = [[0,1],[2,3,4]],
                        AgeBandLimit = 1.0, # irrelevant
                        MandateTax = 0.0,
                        name = 'Baseline',
                        text = 'baseline specification')

# Define alternate specifications for varying the health rating groups
PoorOnlySpec = copy(BaselineSpec)
PoorOnlySpec.HealthStateGroups = [[0],[1,2,3,4]]
PoorOnlySpec.name = 'PoorHealthExcluded'
PoorOnlySpec.text = 'only poor health excluded'
FiveGroupSpec = copy(BaselineSpec)
FiveGroupSpec.HealthStateGroups = [[0],[1],[2],[3],[4]]
FiveGroupSpec.name = 'FullHealthRated'
FiveGroupSpec.text = 'full age-health rating'
AgeRatedSpec = copy(BaselineSpec)
AgeRatedSpec.HealthStateGroups = [[0,1,2,3,4]]
AgeRatedSpec.name = 'OnlyAgeRated'
AgeRatedSpec.text = 'only age rating'
HealthGroupSpecs = [PoorOnlySpec,AgeRatedSpec,FiveGroupSpec]

# Define alternate specifications for varying the age band limit
AgeBandSpecBase = ActuarialSpecification(
                        ActuarialRule = ageRatedActuarialRule,
                        HealthStateGroups = [[0,1,2,3,4]], # irrelevant
                        AgeBandLimit = 10.0,
                        MandateTax = 0.0,
                        name = 'AgeBand10x',
                        text = 'age band limit 10x')
AgeBandSpecs = []
AgeBandLimits = [10.,9.0,8.0,7.0,6.0,5.0,4.0,3.0,2.0]
for AgeBandLimit in AgeBandLimits:
    NewSpec = copy(AgeBandSpecBase)
    NewSpec.AgeBandLimit= AgeBandLimit
    NewSpec.name = 'AgeBand' + str(int(AgeBandLimit)) + 'x'
    NewSpec.text = 'age band limit ' + str(int(AgeBandLimit)) + 'x'
    AgeBandSpecs.append(NewSpec)
FlatSpec = copy(AgeBandSpecBase)
FlatSpec.ActuarialRule = flatActuarialRule
FlatSpec.AgeBandLimit = 1.0 # irrelevant
FlatSpec.name = 'FlatRule'
FlatSpec.text = 'flat pricing'
AgeBandSpecs.append(FlatSpec)

# Define alternate specifications for varying the individual mandate tax
MandateSpecBase = ActuarialSpecification(
                        ActuarialRule = ageRatedActuarialRule,
                        HealthStateGroups = [[0,1,2,3,4]], # irrelevant
                        AgeBandLimit = 3.0,
                        MandateTax = 0.0,
                        name = 'MandateBaseline',
                        text = 'individual mandate 0%')
MandateTaxRates = [0.005,0.010,0.015,0.020,0.025,0.030,0.035,0.04,0.045,0.050,0.055,0.060]
MandateSpecs = []
for MandateTaxRate in MandateTaxRates:
    NewSpec = copy(MandateSpecBase)
    NewSpec.MandateTax = MandateTaxRate
    NewSpec.name = 'Mandate' + str(int(MandateTaxRate*1000))
    NewSpec.text = 'individual mandate ' +  '%.1f' % (MandateTaxRate*100) + '%'
    MandateSpecs.append(NewSpec)
        
    

if __name__ == '__main__':
    import DynInsSelParameters as Params
    
    from time import clock
    
    mystr = lambda number : "{:.4f}".format(number)
    
    # Choose which experiments to work on
    do_health_groups = False
    do_age_bands = False
    do_mandate_tax = True
    
    # Choose what kind of work to do
    run_experiments = False
    make_figures = True
    
    if do_health_groups:
        if run_experiments:
            # Run the health groups experiments
            t_start = clock()
            MyMarket = runCounterfactual(Params.test_param_vec,BaselineSpec,HealthGroupSpecs)
            t_end = clock()
            print('Health groups counterfactual experiment took ' + mystr(t_end-t_start) + ' seconds.')
        
        if make_figures:
            # Make figures for the health groups experiments
            for specification in HealthGroupSpecs:
                makeCounterfactualFigures(specification)
                
    
    if do_age_bands:
        if run_experiments:
            # Run the age band limits experiments
            t_start = clock()
            MyMarket = runCounterfactual(Params.test_param_vec,BaselineSpec,AgeBandSpecs)
            t_end = clock()
            print('Age band limit counterfactual experiment took ' + mystr(t_end-t_start) + ' seconds.')
            
        if make_figures:
            # Make figures for the age bands experiments
            for specification in AgeBandSpecs:
                makeCounterfactualFigures(specification)
    
    if do_mandate_tax:
        if run_experiments:
            # Run the individual mandate experiments
            t_start = clock()
            MyMarket = runCounterfactual(Params.test_param_vec,BaselineSpec,MandateSpecs)
            t_end = clock()
            print('Individual mandate counterfactual experiment took ' + mystr(t_end-t_start) + ' seconds.')
            
        if make_figures:
            # Make figures for the individual mandate experiments
            for specification in MandateSpecs:
                makeCounterfactualFigures(specification)
 
    
    