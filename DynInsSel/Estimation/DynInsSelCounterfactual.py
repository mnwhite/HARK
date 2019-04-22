'''
This module runs counterfactual experiments on the structure of insurance market rules.
'''

import sys 
sys.path.insert(0,'../../')

import numpy as np
import csv
from copy import copy
from DynInsSelEstimation import makeMarketFromParams
from ActuarialRules import PolicySpecification, BaselinePolicySpec, ageHealthRatedActuarialRule, ageRatedActuarialRule
from SubsidyFuncs import NullSubsidyFuncs, makeACAstyleSubsidyPolicy
from HARKutilities import getPercentiles, kernelRegression
from HARKparallel import multiThreadCommands, multiThreadCommandsFake
import matplotlib.pyplot as plt


def runCounterfactual(Parameters,Baseline,Counterfactuals,PremiumLim):
    '''
    Runs a counterfactual experiment comparing individuals' welfare under two
    or more policies.  Constructs a market for the agents to live in, and solves
    it under the "before" rule, generating distribution of individual states.
    It then evaluates the value function for each of the simulated agents, saving
    it as an array.  It then solves the market under the "after" policies, keeping
    the value function(s) again.  For each simulated agent state from the "before"
    world, it does a search for the pLvl_comp where the agent is indifferent be-
    tween the rules: vAfter(mLvl,pLvl_comp,h) = vBefore(mLvl,pLvl,h).  Returns the
    market itself as output, and saves counterfactual data to a CSV file.
    
    Parameters
    ----------
    Parameters : np.array
        Array of 33 structural parameters, just like objectiveFunction.
    Baseline : PolicySpecification
        Specification of the actuarial rule for the "before" state.
    Counterfactuals : [PolicySpecification]
        Specification(s) of the actuarial rule for the counterfactual world.
    PremiumLim : [float,float]
        Vertical axis limit for counterfactual premium-by-age-health figure.
        
    Returns
    -------
    ThisMarket : DynInsSelMarket
        Market that was used to run the counterfactual experiment.
    '''
    FigsDir = '../CounterfactualFigs/'
    
    EvalType  = 3  # Number of times to do a static search for eqbm premiums
    InsChoice = 1  # Extent of insurance choice
    TestPremiums = True # Whether to start with the test premium level
    
    if TestPremiums:
        ESIpremiums = np.array([0.3000, 0.0, 0.0, 0.0, 0.0])
    else:
        ESIpremiums = Params.PremiumsLast
    IMIpremiums_init = Params.IMIpremiums
    
    ContractCounts = [0,1,5] # plus one
    ESIpremiums_init_short = np.concatenate((np.array([0.]),ESIpremiums[0:ContractCounts[InsChoice]]))
    ESIpremiums_init = np.tile(np.reshape(ESIpremiums_init_short,(1,ESIpremiums_init_short.size)),(40,1))
    
    # Make the market, giving it the baseline "before" policy
    ThisMarket = makeMarketFromParams(Parameters,Baseline,IMIpremiums_init,ESIpremiums_init,InsChoice)
    ThisMarket.ESIpremiums = ESIpremiums_init_short
    multiThreadCommandsFake(ThisMarket.agents,['update()','makeShockHistory()'])
    ThisMarket.getIncomeQuintiles()
    multiThreadCommandsFake(ThisMarket.agents,['makeIncBoolArray()'])
    
    # Solve the market with the "before" rule
    ThisMarket.max_loops = EvalType
    ThisMarket.solve()

    # Extract history arrays of m,p,h,v for all agents in the "before" market
    multiThreadCommandsFake(ThisMarket.agents,['makevHistArray()'])
    vHistBefore = np.concatenate([this_type.vNow_hist for this_type in ThisMarket.agents],axis=1)
    mHistBefore = np.concatenate([this_type.mLvlNow_hist for this_type in ThisMarket.agents],axis=1)
    pHist = np.concatenate([this_type.pLvlHist for this_type in ThisMarket.agents],axis=1)
    hBoolArray = np.concatenate([this_type.HealthBoolArray for this_type in ThisMarket.agents],axis=1)
    MrkvHistArray = np.concatenate([this_type.MrkvHist for this_type in ThisMarket.agents],axis=1)
    hHist = -np.ones_like(vHistBefore,dtype=int)
    OfferHist = copy(hHist)
    Alive = MrkvHistArray >= 0
    OfferHist[Alive] = 0
    OfferHist[np.logical_and(Alive,MrkvHistArray >= 5)] = 1
    iHistBefore = np.concatenate([this_type.ContractNow_hist for this_type in ThisMarket.agents],axis=1) > 0
    for h in range(hBoolArray.shape[2]):
        these = hBoolArray[:,:,h]
        hHist[these] = h
    T = ThisMarket.agents[0].T_sim
    N = mHistBefore.shape[1]
    ageHist = np.tile(np.reshape(np.arange(T),(T,1)),(1,N))
    
    # Make a plot of premiums by age and health state
    PremiumsBefore = copy(ThisMarket.IMIpremiums)
    if len(PremiumsBefore.shape) == 1:
        plt.plot([25,64],[10*PremiumsBefore[1],10*PremiumsBefore[1]],'-b')
    elif len(PremiumsBefore.shape) == 2:
        plt.plot(np.arange(25,65),10*PremiumsBefore[:,1],'-b')
    elif len(PremiumsBefore.shape) == 3:
        for g in range(0,PremiumsBefore.shape[1]):
            plt.plot(np.arange(25,65),10*PremiumsBefore[:,g,1],'-')
    plt.xlabel('Age')
    plt.ylabel('Annual premium (thousands of USD)')
    plt.ylim(PremiumLim)
    plt.title('Premiums by age in baseline scenario')
    plt.savefig(FigsDir + 'PremiumsBaseline.pdf')
    plt.show()
    
    for Counterfactual in Counterfactuals:
        # Now give the market the "after" rule, and have it update premiums using the old distribution, then solve it properly
        ThisMarket.updatePolicy(Counterfactual)
        ThisMarket.makeHistory()
        ThisMarket.updateDynamics() # make sure agents actually get the new premiums
        ThisMarket.solve()
                
        # Make a plot of premiums by age and health state
        PremiumsAfter = copy(ThisMarket.IMIpremiums)
        if len(PremiumsAfter.shape) == 1:
            plt.plot([25,64],[10*PremiumsAfter[1],10*PremiumsAfter[1]],'-b')
        elif len(PremiumsAfter.shape) == 2:
            plt.plot(np.arange(25,65),10*PremiumsAfter[:,1],'-b')
        elif len(PremiumsAfter.shape) == 3:
            for g in range(0,PremiumsAfter.shape[1]):
                plt.plot(np.arange(25,65),10*PremiumsAfter[:,g,1],'-')
        plt.xlabel('Age')
        plt.ylabel('Annual premium (thousands of USD)')
        plt.title('Premiums by age, ' + Counterfactual.text)
        plt.ylim(PremiumLim)
        plt.savefig(FigsDir + 'Premiums' + Counterfactual.name + '.pdf')
        plt.show()
        
        # Replace the simulated histories of the after market with those of the before market
        K = 0 # Running count of position in big arrays
        for this_type in ThisMarket.agents:
            this_type.mLvlNow_hist = mHistBefore[:,K:(K+this_type.AgentCount)]
            this_type.vTarg_hist = vHistBefore[:,K:(K+this_type.AgentCount)]
            K += this_type.AgentCount
            # Each type should have the same pLvlHist and HealthBoolArray before and after (don't need to change it)
            
        # Find compensating variation for each simulated individual, in terms of pLvl (and extract the results)
        multiThreadCommands(ThisMarket.agents,['findCompensatingpLvl()'])
        pComp_all = np.concatenate([this_type.pCompHist for this_type in ThisMarket.agents],axis=1)
        Invalid_all = np.concatenate([this_type.pComp_invalid for this_type in ThisMarket.agents],axis=1)
        iHistAfter = np.concatenate([this_type.ContractNow_hist for this_type in ThisMarket.agents],axis=1) > 0
        
        # Reshape the useable data into 1D arrays and combine
        which = np.logical_and(hHist >= 0,ageHist < 40)
        mLvl = mHistBefore[which]
        pLvl = pHist[which]
        health = hHist[which]
        offer = OfferHist[which]
        age = ageHist[which]
        pComp = pComp_all[which]
        Invalid = Invalid_all[which]
        iBefore = iHistBefore[which]
        iAfter = iHistAfter[which]
        counterfactual_data = np.vstack([age,health,offer,mLvl,pLvl,pComp,Invalid,iBefore,iAfter]).transpose()
        
        # Write the counterfactual results to a CSV file
        VarNames = ['age','health','offer','mLvl','pLvl','pComp','Invalid','iBefore','iAfter']
        with open('../Results/' + Counterfactual.name + 'Data.txt.','wb') as f:
            my_writer = csv.writer(f, delimiter = '\t')
            my_writer.writerow(VarNames)
            for i in range(counterfactual_data.shape[0]):
                X = counterfactual_data[i,:]
                this_row = ['%.0f' % X[0], '%.0f' % X[1], '%.0f' % X[2], '%.4f' % X[3], '%.4f' % X[4], '%.4f' % X[5], '%.0f' % X[6], '%.0f' % X[7], '%.0f' % X[8]]
                my_writer.writerow(this_row)
            f.close()
            
    return ThisMarket


def makeCounterfactualFigures(specification,AgeHealthLim,AgeIncomeLim,IncomeHealthLim):
    '''
    Produces many figures to graphically represent the results of a counterfactual
    experiment and saves them to the folder ../CounterfactualFigures.
    
    Parameters
    ----------
    specification : ActuarialSpecification
        Counterfactual specification whose figures are to be produced.
    AgeHealthLim : [float,float]
        Vertical axis limits for the age-health WTP plot.
    IncomeAgeLim : [float,float]
        Vertical axis limits for the income-age WTP plot.
    IncomeHealthLim : [float,float]
        Vertical axis limits for the income-health WTP plot.
        
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
    offer = np.zeros(N,dtype=int)
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
        offer[i] = int(float(all_data[j][2]))
        mLvl[i] = float(all_data[j][3])
        pLvl[i] = float(all_data[j][4])
        pComp[i] = float(all_data[j][5])
        invalid[i] = bool(float(all_data[j][6]))
        iBefore[i] = bool(float(all_data[j][7]))
        iAfter[i] = bool(float(all_data[j][8]))  
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
    IMIinsuredRateByAge = np.zeros(T)
    IMIinsuredRateByAgeHealth = np.zeros((T,5))
    for t in range(T):
        these = np.logical_and(age == t, np.logical_not(offer))
        IMIinsuredRateByAge[t] = float(np.sum(iBefore[these]))/float(np.sum(these))
        for h in range(5):
            those = np.logical_and(these, health == h)
            IMIinsuredRateByAgeHealth[t,h] = float(np.sum(iBefore[those]))/float(np.sum(those))
    plt.plot(AgeVec,IMIinsuredRateByAge,'--k')
    plt.plot(AgeVec,IMIinsuredRateByAgeHealth,'-')
    plt.xlabel('Age')
    plt.ylabel('Insured rate by health status')
    plt.title('Individual market insured rate, baseline scenario')
    plt.ylim(0.,1.)
    plt.savefig(FigsDir + 'InsuredRateHealthBaseline.pdf')
    plt.show()
    
    # Make a plot of insured rate by age (and income) in the before scenario
    IMIinsuredRateByAgeIncome = np.zeros((T,5))
    for t in range(T):
        these = np.logical_and(age == t, np.logical_not(offer))
        p_temp = pLvl[these]
        i_temp = iBefore[these]
        quintile_cuts = getPercentiles(p_temp,percentiles=[0.2,0.4,0.6,0.8])
        quintiles = np.zeros(np.sum(these),dtype=int)
        for i in range(4):
            quintiles[p_temp > quintile_cuts[i]] += 1
        for i in range(5):
            those = quintiles==i
            IMIinsuredRateByAgeIncome[t,i] = float(np.sum(i_temp[those]))/float(np.sum(those))
    plt.plot(AgeVec,IMIinsuredRateByAge,'--k')
    plt.plot(AgeVec,IMIinsuredRateByAgeIncome,'-')
    plt.xlabel('Age')
    plt.ylabel('Insured rate by income quintile')
    plt.title('Individual market insured rate, baseline scenario')
    plt.ylim(0.,1.)
    plt.savefig(FigsDir + 'InsuredRateIncomeBaseline.pdf')
    plt.show()
    
    # Make a plot of insured rate by age (and health) in the after scenario
    IMIinsuredRateByAge = np.zeros(T)
    IMIinsuredRateByAgeHealth = np.zeros((T,5))
    for t in range(T):
        these = np.logical_and(age == t, np.logical_not(offer))
        IMIinsuredRateByAge[t] = float(np.sum(iAfter[these]))/float(np.sum(these))
        for h in range(5):
            those = np.logical_and(these, health == h)
            IMIinsuredRateByAgeHealth[t,h] = float(np.sum(iAfter[those]))/float(np.sum(those))
    plt.plot(AgeVec,IMIinsuredRateByAge,'--k')
    plt.plot(AgeVec,IMIinsuredRateByAgeHealth,'-')
    plt.xlabel('Age')
    plt.ylabel('Insured rate by health status')
    plt.title('Individual market insured rate, ' + specification.text)
    plt.ylim(0.,1.)
    plt.savefig(FigsDir + 'InsuredRateHealth' + specification.name + '.pdf')
    plt.show()
    
    # Make a plot of insured rate by age (and income) in the after scenario
    IMIinsuredRateByAgeIncome = np.zeros((T,5))
    for t in range(T):
        these = np.logical_and(age == t, np.logical_not(offer))
        p_temp = pLvl[these]
        i_temp = iAfter[these]
        quintile_cuts = getPercentiles(p_temp,percentiles=[0.2,0.4,0.6,0.8])
        quintiles = np.zeros(np.sum(these),dtype=int)
        for i in range(4):
            quintiles[p_temp > quintile_cuts[i]] += 1
        for i in range(5):
            those = quintiles==i
            IMIinsuredRateByAgeIncome[t,i] = float(np.sum(i_temp[those]))/float(np.sum(those))
    plt.plot(AgeVec,IMIinsuredRateByAge,'--k')
    plt.plot(AgeVec,IMIinsuredRateByAgeIncome,'-')
    plt.xlabel('Age')
    plt.ylabel('Insured rate by income quintile')
    plt.title('Individual market insured rate, ' + specification.text)
    plt.ylim(0.,1.)
    plt.savefig(FigsDir + 'InsuredRateIncome' + specification.name + '.pdf')
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
    plt.legend(labels=['Poor','Fair','Good','Very good','Excellent'],loc=2)
    plt.ylim(AgeHealthLim)
    plt.savefig(FigsDir + 'WTPbyAgeHealth' + specification.name + '.pdf')
    plt.show()
    
    # Make a plot of kernel regression of willingness to pay by age and income quintile
    WTPmeanByAgeIncome = np.zeros((T,5)) + np.nan
    for t in range(T):
        these = np.logical_and(age == t, valid)
        p_temp = pLvl[these]
        WTP_temp = WTP[these]
        quintile_cuts = getPercentiles(p_temp,percentiles=[0.2,0.4,0.6,0.8])
        quintiles = np.zeros(np.sum(these),dtype=int)
        for i in range(4):
            quintiles[p_temp > quintile_cuts[i]] += 1
        for i in range(5):
            those = quintiles==i
            WTPmeanByAgeIncome[t,i] = np.mean(WTP_temp[those])
    plt.plot(AgeVec,WTPmeanByAgeIncome*100,'-')
    plt.xlabel('Age')
    plt.ylabel('Willingness-to-pay (% of permanent income)')
    plt.title('Mean willingness-to-pay by age and income, ' + specification.text)
    plt.legend(labels=['Poorest quintile','Second quintile','Third quintile','Fourth quintile','Richest quintile'],loc=2)
    plt.ylim(AgeIncomeLim)
    plt.savefig(FigsDir + 'WTPbyAgeIncome' + specification.name + '.pdf')
    plt.show()
    
    # Make a plot of kernel regression of willingness to pay by age and income quintile
    WTPmeanByAgeOffer = np.zeros((T,3)) + np.nan
    for t in range(T):
        these = np.logical_and(age == t, valid)
        WTP_temp = WTP[these]
        for o in range(2):
            those = offer[these] == o
            WTPmeanByAgeOffer[t,o] = np.mean(WTP_temp[those])
        WTPmeanByAgeOffer[t,2] = np.mean(WTP_temp)
    plt.plot(AgeVec,WTPmeanByAgeOffer[:,0]*100,'-b')
    plt.plot(AgeVec,WTPmeanByAgeOffer[:,1]*100,'-r')
    plt.plot(AgeVec,WTPmeanByAgeOffer[:,2]*100,'-k')
    plt.xlabel('Age')
    plt.ylabel('Willingness-to-pay (% of permanent income)')
    plt.title('Mean willingness-to-pay by age and ESI status, ' + specification.text)
    plt.legend(labels=['Not offered ESI','Offered ESI','Overall'],loc=2)
    plt.ylim(AgeIncomeLim)
    plt.savefig(FigsDir + 'WTPbyAgeOffer' + specification.name + '.pdf')
    plt.show()
    
#    for a in range(4):
#        age_min = a*10
#        age_max = age_min+10
#        these = np.logical_and(np.logical_and(age >= age_min,age < age_max), valid)
#        p_temp = pLvl[these]
#        cuts = getPercentiles(p_temp,percentiles=[0.01,0.99])
#        f = kernelRegression(p_temp,WTP[these],bot=cuts[0],top=20.,N=200,h=0.5)
#        P = np.linspace(cuts[0],20.,200)
#        plt.plot(P*10,f(P)*100,'-')
#    plt.xlabel('Permanent income level (thousands of USD)')
#    plt.ylabel('Willingness-to-pay (% of permanent income)')
#    plt.title('Mean willingness-to-pay by income and age, ' + specification.text)
#    plt.legend(labels=['Age 25-34','Age 35-44','Age 45-54','Age 55-64'],loc=1)
#    plt.ylim(IncomeAgeLim)
#    plt.savefig(FigsDir + 'WTPbyAgeIncome' + specification.name + '.pdf')
#    plt.show()
    
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
    plt.legend(labels=['Poor','Fair','Good','Very good','Excellent'],loc=1)
    plt.ylim(IncomeHealthLim)
    plt.savefig(FigsDir + 'WTPbyIncomeHealth' + specification.name + '.pdf')
    plt.show()

        

###############################################################################
###############################################################################


# Baseline specification is imported from ActuarialRules.py, same as in estimation
BaselineSpec = BaselinePolicySpec

# Define alternate specifications for varying the health rating groups
OnlyPoorHealthSpec = copy(BaselineSpec)
OnlyPoorHealthSpec.HealthGroups = [[0],[1,2,3,4]]
OnlyPoorHealthSpec.name = 'PoorHealthExcluded'
OnlyPoorHealthSpec.text = 'only poor health separated'

FiveGroupSpec = copy(BaselineSpec)
FiveGroupSpec.HealthGroups = [[0],[1],[2],[3],[4]]
FiveGroupSpec.ExcludedGroups = [False,False,False,False,False]
FiveGroupSpec.name = 'FullHealthRated'
FiveGroupSpec.text = 'full age-health rating'

AgeRatedSpec = copy(BaselineSpec)
AgeRatedSpec.HealthGroups = [[0,1,2,3,4]]
AgeRatedSpec.ExcludedGroups = [False]
AgeRatedSpec.name = 'OnlyAgeRated'
AgeRatedSpec.text = 'only age rating'

HealthGroupSpecs = [OnlyPoorHealthSpec,FiveGroupSpec,AgeRatedSpec]

# Define alternate specifications for varying the age band limit
AgeBandSpecBase = PolicySpecification(
                        SubsidyFunc=NullSubsidyFuncs,
                        ActuarialRule = ageRatedActuarialRule,
                        HealthGroups = [[0,1,2,3,4]], # irrelevant
                        ExcludedGroups = [False], # irrelevant
                        AgeBandLimit = 10.0,
                        MandateTaxRate = 0.0,
                        MandateFloor = 0.0,
                        MandateForESI = False,
                        name = 'AgeBand10x',
                        text = 'age band limit 10x')
AgeBandSpecs = [AgeRatedSpec]
AgeBandLimits = [5.0,4.8,4.6,4.4,4.2,4.0,3.8,3.6,3.4,3.2,3.0,2.8,2.6,2.4,2.2,2.0,1.8,1.6,1.4,1.2,1.0]
for AgeBandLimit in AgeBandLimits:
    NewSpec = copy(AgeBandSpecBase)
    NewSpec.AgeBandLimit= AgeBandLimit
    NewSpec.name = 'AgeBand' + str(int(AgeBandLimit*10)) + 'x'
    NewSpec.text = 'age band limit ' + '%.1f' % AgeBandLimit + 'x'
    AgeBandSpecs.append(NewSpec)


# Define alternate specifications for varying the individual mandate tax
MandateSpecBase = PolicySpecification(
                        ActuarialRule = ageRatedActuarialRule,
                        SubsidyFunc=NullSubsidyFuncs,
                        HealthGroups = [[0,1,2,3,4]], # irrelevant
                        ExcludedGroups = [False], # irrelevant
                        AgeBandLimit = 3.0,
                        MandateTaxRate = 0.0,
                        MandateFloor = 0.0,
                        MandateForESI = False,
                        name = 'MandateBaseline',
                        text = 'individual mandate 0%')
MandateTaxRates = [0.000,0.005,0.010,0.015,0.020,0.025,0.030,0.035,0.04,0.045,0.050,0.055,0.060,0.065,0.070,0.075,0.080,0.085,0.090,0.095,0.100]
MandateSpecs = [AgeRatedSpec]
for MandateTaxRate in MandateTaxRates:
    NewSpec = copy(MandateSpecBase)
    NewSpec.MandateTaxRate = MandateTaxRate
    NewSpec.name = 'Mandate' + str(int(MandateTaxRate*1000))
    NewSpec.text = 'individual mandate ' +  '%.1f' % (MandateTaxRate*100) + '%'
    MandateSpecs.append(NewSpec)
    
    
# Define alternate specifications for ACA-style subsidies
ACAspecBase = PolicySpecification(
                        ActuarialRule = ageRatedActuarialRule,
                        SubsidyFunc=None, # will be replaced below
                        HealthGroups = [[0,1,2,3,4]], # irrelevant
                        ExcludedGroups = [False], # irrelevant
                        AgeBandLimit = 3.0,
                        MandateTaxRate = 0.025,
                        MandateFloor = 0.07,
                        MandateForESI = False,
                        name = 'ACAbaseline',
                        text = '400% FPL eligibility cap')
FPLcutoffs = [3.00,3.25,3.50,3.75,4.00,4.25,4.50,4.75,5.0,5.25,5.50,5.75,6.00]
ACAaltFPLcutoffSpecs = [AgeRatedSpec]
for FPLcutoff in FPLcutoffs:
    NewSpec = copy(ACAspecBase)
    NewSubsidyFunc = makeACAstyleSubsidyPolicy(0.095,FPLcutoff)
    NewSpec.SubsidyFunc = NewSubsidyFunc
    NewSpec.name = 'ACAcutoff' + str(int(FPLcutoff*100))
    NewSpec.text = str(int(FPLcutoff*100)) + '% FPL eligibility cap'
    ACAaltFPLcutoffSpecs.append(NewSpec)
    
MaxOOPpcts = [0.080,0.085,0.090,0.095,0.100,0.105,0.110,0.115,0.120]
ACAaltMaxOOPpctSpecs = [AgeRatedSpec]
for MaxOOPpct in MaxOOPpcts:
    NewSpec = copy(ACAspecBase)
    NewSubsidyFunc = makeACAstyleSubsidyPolicy(MaxOOPpct,4.0)
    NewSpec.SubsidyFunc = NewSubsidyFunc
    NewSpec.name = 'ACAmaxOOP' + str(int(MaxOOPpct*1000))
    NewSpec.text = 'max OOP premium: ' + str(int(MaxOOPpct*100)) + '% income' 
    ACAaltMaxOOPpctSpecs.append(NewSpec)
        
    

if __name__ == '__main__':
    import DynInsSelParameters as Params
    
    from time import clock
    
    mystr = lambda number : "{:.4f}".format(number)
    
    # Choose which experiments to work on
    do_health_groups = False
    do_age_bands = False
    do_mandate_tax = False
    do_eligibility_cutoff = False
    do_max_OOP_prem = True
    
    # Choose what kind of work to do
    run_experiments = True
    make_figures = True
    
    if do_health_groups:
        if run_experiments:
            # Run the health groups experiments
            t_start = clock()
            MyMarket = runCounterfactual(Params.test_param_vec,BaselineSpec,HealthGroupSpecs,[0.,20.])
            t_end = clock()
            print('Health groups counterfactual experiment took ' + mystr(t_end-t_start) + ' seconds.')
        
        if make_figures:
            # Make figures for the health groups experiments
            for specification in HealthGroupSpecs:
                makeCounterfactualFigures(specification,[-5,25],[-5,25],[-5,25])
                
    
    if do_age_bands:
        if run_experiments:
            # Run the age band limits experiments
            t_start = clock()
            MyMarket = runCounterfactual(Params.test_param_vec,BaselineSpec,AgeBandSpecs,[0.,20.])
            t_end = clock()
            print('Age band limit counterfactual experiment took ' + mystr(t_end-t_start) + ' seconds.')
            
        if make_figures:
            # Make figures for the age bands experiments
            for specification in AgeBandSpecs[1:]:
                makeCounterfactualFigures(specification,[-20,35],[-11,25],[-5,35])

    
    if do_mandate_tax:
        if run_experiments:
            # Run the individual mandate experiments
            t_start = clock()
            MyMarket = runCounterfactual(Params.test_param_vec,BaselineSpec,MandateSpecs,[0.,20.])
            t_end = clock()
            print('Individual mandate counterfactual experiment took ' + mystr(t_end-t_start) + ' seconds.')
            
        if make_figures:
            # Make figures for the individual mandate experiments
            for specification in MandateSpecs:
                makeCounterfactualFigures(specification,[-10,30],[-15,20],[-5,35])

                
    if do_eligibility_cutoff:
        if run_experiments:
            # Run the subsidy eligibility (FPL pct) experiments
            t_start = clock()
            MyMarket = runCounterfactual(Params.test_param_vec,BaselineSpec,ACAaltFPLcutoffSpecs,[0.,20.])
            t_end = clock()
            print('Eligibility cutoff counterfactual experiment took ' + mystr(t_end-t_start) + ' seconds.')
            
        if make_figures:
            # Make figures for the subsidy eligibility (FPL pct) experiments
            for specification in ACAaltFPLcutoffSpecs:
                makeCounterfactualFigures(specification,[-10,30],[-15,20],[-5,35])
                
                
    if do_max_OOP_prem:
        if run_experiments:
            # Run the maximum OOP premium experiments
            t_start = clock()
            MyMarket = runCounterfactual(Params.test_param_vec,BaselineSpec,ACAaltMaxOOPpctSpecs,[0.,20.])
            t_end = clock()
            print('Eligibility cutoff counterfactual experiment took ' + mystr(t_end-t_start) + ' seconds.')
            
        if make_figures:
            # Make figures for the maximum OOP premium (FPL pct) experiments
            for specification in ACAaltMaxOOPpctSpecs:
                makeCounterfactualFigures(specification,[-10,30],[-15,20],[-5,35])             
    
    