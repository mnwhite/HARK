'''
This module runs counterfactual experiments on the structure of insurance market rules.
'''

import sys 
sys.path.insert(0,'../../')

import numpy as np
import csv
from copy import copy
from DynInsSelEstimation import makeMarketFromParams
from ActuarialRules import PolicySpecification, PreACAbaselineSpec, PostACAbaselineSpec, generalIMIactuarialRule
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
        print('Now solving a counterfactual policy named ' + str(Counterfactual.text) + '.')
        
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
            print('Wrote counterfactual results to ' + Counterfactual.name + 'Data.txt.')
            
    return ThisMarket


def makeCounterfactualFigures(specification,AgeHealthLim,AgeIncomeLim,OfferAgeLim,IncomeHealthLim):
    '''
    Produces many figures to graphically represent the results of a counterfactual
    experiment and saves them to the folder ../CounterfactualFigures.
    
    Parameters
    ----------
    specification : ActuarialSpecification
        Counterfactual specification whose figures are to be produced.
    AgeHealthLim : [float,float]
        Vertical axis limits for the age-health WTP plot.
    AgeIncomeLim : [float,float]
        Vertical axis limits for the income-age WTP plot.
    OfferAgeLim : [float,float]
        Vertical axis limits for the offer-age WTP plot
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
        these = age == t
        p_temp = pLvl[these]
        i_temp = iAfter[these]
        quintile_cuts = getPercentiles(p_temp,percentiles=[0.2,0.4,0.6,0.8])
        quintiles = np.zeros(np.sum(these),dtype=int)
        for i in range(4):
            quintiles[p_temp > quintile_cuts[i]] += 1
        for i in range(5):
            those = np.logical_and(quintiles==i, np.logical_not(offer))
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
    
    # Make a plot of mean willingness to pay by age and income quintile
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
    
    # Make a plot of mean willingness to pay by age and ESI offer status
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
    plt.ylim(OfferAgeLim)
    plt.savefig(FigsDir + 'WTPbyAgeOffer' + specification.name + '.pdf')
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
    plt.legend(labels=['Poor','Fair','Good','Very good','Excellent'],loc=1)
    plt.ylim(IncomeHealthLim)
    plt.savefig(FigsDir + 'WTPbyIncomeHealth' + specification.name + '.pdf')
    plt.show()

        

def makeCrossPolicyFigures(name,specifications,AgeHealthLim,AgeIncomeLim,AgeOfferLim):
    '''
    Produces many figures to graphically represent the results of a counterfactual
    experiment and saves them to the folder ../CounterfactualFigures.
    
    Parameters
    ----------
    name : str
        Filename prefix for this set of specifications
    specifications : [ActuarialSpecification]
        Counterfactual specification whose figures are to be produced.
    AgeHealthLim : [float,float]
        Vertical axis limits for the age-health WTP plot.
    AgeIncomeLim : [float,float]
        Vertical axis limits for the income-age WTP plot.
    AgeOfferLim : [float,float]
        Vertical axis limits for the offer-age WTP plot
        
    Returns
    -------
    None
    '''
    T = 40
    P = len(specifications)
    
    # Initialize arrays to hold counterfactual results
    IMIinsuredRateByAge = np.zeros((P,T))
    IMIinsuredRateByAgeGoodHealth = np.zeros((P,T))
    IMIinsuredRateByAgeBadHealth = np.zeros((P,T))
    IMIinsuredRateByAgeHighInc = np.zeros((P,T))
    IMIinsuredRateByAgeMidInc = np.zeros((P,T))
    IMIinsuredRateByAgeLowInc = np.zeros((P,T))
    WTPmeanByAge = np.zeros((P,T))
    WTPmeanByAgeGoodHealth = np.zeros((P,T))
    WTPmeanByAgeBadHealth = np.zeros((P,T))
    WTPmeanByAgeHighInc = np.zeros((P,T))
    WTPmeanByAgeMidInc = np.zeros((P,T))
    WTPmeanByAgeLowInc = np.zeros((P,T))
    WTPmeanByAgeIMI = np.zeros((P,T))
    WTPmeanByAgeESI = np.zeros((P,T))
    label_list = []
    
    for p in range(P):
        specification = specifications[p]
        print('Processing the specification labeled ' + specification.text + '...')
        label_list.append(specification.text)
        
        with open('../Results/' + specification.name + 'Data.txt','r') as f:
            my_reader = csv.reader(f, delimiter = '\t')
            all_data = list(my_reader)
        FigsDir = '../CounterfactualFigs/'
        
        # Initialize data arrays
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
        
        # Read in the data
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
        
        good_health = health >= 3
        bad_health = health < 3
        
        # Calculate IMI insured rate for this policy by age and health
        for t in range(T):
            these = np.logical_and(age == t, np.logical_not(offer))
            IMIinsuredRateByAge[p,t] = float(np.sum(iAfter[these]))/float(np.sum(these))
            those = np.logical_and(these, good_health)
            IMIinsuredRateByAgeGoodHealth[p,t] = float(np.sum(iAfter[those]))/float(np.sum(those))
            those = np.logical_and(these, bad_health)
            IMIinsuredRateByAgeBadHealth[p,t] = float(np.sum(iAfter[those]))/float(np.sum(those))
        
        # Calculate mean WTP by age, health, and offer status
        for t in range(T):
            these = np.logical_and(age == t, valid)
            WTPmeanByAge[p,t] = np.mean(WTP[these])
            those = np.logical_and(these, good_health)
            WTPmeanByAgeGoodHealth[p,t] = np.mean(WTP[those])
            those = np.logical_and(these, bad_health)
            WTPmeanByAgeBadHealth[p,t] = np.mean(WTP[those])
            those = np.logical_and(these, offer)
            WTPmeanByAgeESI[p,t] = np.mean(WTP[those])
            those = np.logical_and(these, np.logical_not(offer))
            WTPmeanByAgeIMI[p,t] = np.mean(WTP[those])
            
        # Calculate IMI insured rate and WTP for this policy by age and income       
        for t in range(T):
            these = age == t
            p_temp = pLvl[these]
            i_temp = iAfter[these]
            WTP_temp = WTP[these]
            group_cuts = getPercentiles(p_temp,percentiles=[0.2,0.8])
            group = np.zeros(np.sum(these),dtype=int)
            for i in range(2):
                group[p_temp > group_cuts[i]] += 1
            IMI_temp = np.logical_not(offer[these])
            those = np.logical_and(group == 0, IMI_temp)
            IMIinsuredRateByAgeLowInc[p,t] = float(np.sum(i_temp[those]))/float(np.sum(those))
            those = np.logical_and(group == 1, IMI_temp)
            IMIinsuredRateByAgeMidInc[p,t] = float(np.sum(i_temp[those]))/float(np.sum(those))
            those = np.logical_and(group == 2, IMI_temp)
            IMIinsuredRateByAgeHighInc[p,t] = float(np.sum(i_temp[those]))/float(np.sum(those))
            
            group = np.zeros(np.sum(these),dtype=int)
            for i in range(2):
                group[p_temp > group_cuts[i]] += 1
            group[np.logical_not(valid[these])] = -1
            those = group == 0
            WTPmeanByAgeLowInc[p,t] = np.mean(WTP_temp[those])
            those = group == 1
            WTPmeanByAgeMidInc[p,t] = np.mean(WTP_temp[those])
            those = group == 2
            WTPmeanByAgeHighInc[p,t] = np.mean(WTP_temp[those])
            
    
    line_styles = ['-b','-g','-r','-c','-m']
    AgeVec = np.arange(25,65)
    
    plt.figure(figsize=(6.4,5.0))
    
    plt.subplot(3,2,1)
    for p in range(P):
        plt.plot(AgeVec,IMIinsuredRateByAge[p,:], line_styles[p])
    #plt.xlabel('Age')
    plt.ylabel('Overall')
    plt.xlim(25,65)
    plt.ylim(0.,1.)
    plt.xticks([])
    
    plt.subplot(3,2,3)
    for p in range(P):
        plt.plot(AgeVec,IMIinsuredRateByAgeGoodHealth[p,:], line_styles[p])
    #plt.xlabel('Age')
    plt.ylabel('Healthy')
    plt.xlim(25,65)
    plt.ylim(0.,1.)
    plt.xticks([])
    
    plt.subplot(3,2,5)
    for p in range(P):
        plt.plot(AgeVec,IMIinsuredRateByAgeBadHealth[p,:], line_styles[p])
    plt.xlabel('Age')
    plt.ylabel('Unhealthy')
    plt.xlim(25,65)
    plt.ylim(0.,1.)
    plt.legend(labels=label_list,bbox_to_anchor=(0., -0.55, 2.5, .102), loc=10,
           ncol=P, mode="expand")
    
    plt.subplot(3,2,2)
    for p in range(P):
        plt.plot(AgeVec,IMIinsuredRateByAgeHighInc[p,:], line_styles[p])
    #plt.xlabel('Age')
    plt.ylabel('High income')
    plt.xlim(25,65)
    plt.ylim(0.,1.)
    plt.xticks([])
    
    plt.subplot(3,2,4)
    for p in range(P):
        plt.plot(AgeVec,IMIinsuredRateByAgeMidInc[p,:], line_styles[p])
    #plt.xlabel('Age')
    plt.ylabel('Mid income')
    plt.xlim(25,65)
    plt.ylim(0.,1.)
    plt.xticks([])
    
    plt.subplot(3,2,6)
    for p in range(P):
        plt.plot(AgeVec,IMIinsuredRateByAgeLowInc[p,:], line_styles[p])
    plt.xlabel('Age')
    plt.ylabel('Low income')
    plt.xlim(25,65)
    plt.ylim(0.,1.)
    
    plt.suptitle('Individual market insured rate across counterfactual policies',y=1.02)
    plt.tight_layout()
    plt.savefig(FigsDir + name + 'InsuredRateIMI.pdf',bbox_inches='tight')
    plt.show()
    
    
    
    plt.figure(figsize=(6.4,6.6))
    
    plt.subplot(4,2,1)
    for p in range(P):
        plt.plot(AgeVec,WTPmeanByAge[p,:]*100, line_styles[p])
    plt.ylabel('Overall')
    plt.xlim(25,65)
    #plt.ylim(0.,1.)
    plt.xticks([])
    
    plt.subplot(4,2,3)
    for p in range(P):
        plt.plot(AgeVec,WTPmeanByAgeGoodHealth[p,:]*100, line_styles[p])
    plt.ylabel('Healthy')
    plt.xlim(25,65)
    plt.ylim(AgeHealthLim[0],AgeHealthLim[1])
    plt.xticks([])
    
    plt.subplot(4,2,5)
    for p in range(P):
        plt.plot(AgeVec,WTPmeanByAgeBadHealth[p,:]*100, line_styles[p])
    plt.ylabel('Unhealthy')
    plt.xlim(25,65)
    plt.ylim(AgeHealthLim[0],AgeHealthLim[1])
    plt.xticks([])
    
    plt.subplot(4,2,7)
    for p in range(P):
        plt.plot(AgeVec,WTPmeanByAgeIMI[p,:]*100, line_styles[p])
    plt.xlabel('Age')
    plt.ylabel('Not offered ESI')
    plt.xlim(25,65)
    plt.ylim(AgeOfferLim[0],AgeOfferLim[1])
    plt.legend(labels=label_list,bbox_to_anchor=(0., -0.55, 2.5, .102), loc=10,
           ncol=P, mode="expand")
    
    plt.subplot(4,2,2)
    for p in range(P):
        plt.plot(AgeVec,WTPmeanByAgeHighInc[p,:]*100, line_styles[p])
    plt.ylabel('High income')
    plt.xlim(25,65)
    plt.ylim(AgeIncomeLim[0],AgeIncomeLim[1])
    plt.xticks([])
    
    plt.subplot(4,2,4)
    for p in range(P):
        plt.plot(AgeVec,WTPmeanByAgeMidInc[p,:]*100, line_styles[p])
    plt.ylabel('Mid income')
    plt.xlim(25,65)
    plt.ylim(AgeIncomeLim[0],AgeIncomeLim[1])
    plt.xticks([])
    
    plt.subplot(4,2,6)
    for p in range(P):
        plt.plot(AgeVec,WTPmeanByAgeLowInc[p,:]*100, line_styles[p])
    plt.ylabel('Low income')
    plt.xlim(25,65)
    plt.ylim(AgeIncomeLim[0],AgeIncomeLim[1])
    plt.xticks([])
    
    plt.subplot(4,2,8)
    for p in range(P):
        plt.plot(AgeVec,WTPmeanByAgeESI[p,:]*100, line_styles[p])
    plt.xlabel('Age')
    plt.ylabel('Offered ESI')
    plt.xlim(25,65)
    plt.ylim(AgeOfferLim[0],AgeOfferLim[1])
    
    plt.suptitle('Mean WTP (as % of permanent income) for counterfactual policies',y=1.02)
    plt.tight_layout()
    plt.savefig(FigsDir + name + 'WTPmean.pdf',bbox_inches='tight')
    plt.show()
        
        
    


###############################################################################
###############################################################################


# Baseline specification is imported from ActuarialRules.py, same as in estimation
BaselineSpec = PreACAbaselineSpec

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
                        SubsidyFunc = NullSubsidyFuncs,
                        ActuarialRule = generalIMIactuarialRule,
                        HealthGroups = [[0,1,2,3,4]],
                        ExcludedGroups = [False],
                        AgeBandLimit = None, # Will be overwritten below
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
                        ActuarialRule = generalIMIactuarialRule,
                        SubsidyFunc = NullSubsidyFuncs,
                        HealthGroups = [[0,1,2,3,4]],
                        ExcludedGroups = [False],
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
                        ActuarialRule = generalIMIactuarialRule,
                        SubsidyFunc = None, # will be replaced below
                        HealthGroups = [[0,1,2,3,4]],
                        ExcludedGroups = [False],
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
    

# Define a set of policy counterfactuals that one ACA policy feature at a time
AddCommunityRatingSpec = copy(PreACAbaselineSpec)
AddCommunityRatingSpec.HealthGroups = [[0,1,2,3,4,5]]
AddCommunityRatingSpec.ExcludedGroups = [False]

AddAgeRatingLimitSpec = copy(PreACAbaselineSpec)
AddAgeRatingLimitSpec.AgeBandLimit = 3.0

AddIndividualMandateSpec = copy(PreACAbaselineSpec)
AddIndividualMandateSpec.MandateTaxRate = 0.025
AddIndividualMandateSpec.MandateFloor = 0.07

AddIMIsubsidiesSpec = copy(PreACAbaselineSpec)
AddIMIsubsidiesSpec.SubsidyFunc = makeACAstyleSubsidyPolicy(0.095, 4.0)

AddACAfeaturesSpecs = [AddCommunityRatingSpec,AddAgeRatingLimitSpec,AddIndividualMandateSpec,AddIMIsubsidiesSpec]


# Define a set of policy counterfactuals that remove ACA policy feature at a time
DelCommunityRatingSpec = copy(PostACAbaselineSpec)
DelCommunityRatingSpec.HealthGroups = [[0,1],[2,3,4,5]]
DelCommunityRatingSpec.ExcludedGroups = [False,False]

DelAgeRatingLimitSpec = copy(PostACAbaselineSpec)
DelAgeRatingLimitSpec.AgeBandLimit = None

DelIndividualMandateSpec = copy(PostACAbaselineSpec)
DelIndividualMandateSpec.MandateTaxRate = 0.00
DelIndividualMandateSpec.MandateFloor = 0.00

DelIMIsubsidiesSpec = copy(PostACAbaselineSpec)
DelIMIsubsidiesSpec.SubsidyFunc = NullSubsidyFuncs

DelACAfeaturesSpecs = [DelCommunityRatingSpec,DelAgeRatingLimitSpec,DelIndividualMandateSpec,DelIMIsubsidiesSpec]



if __name__ == '__main__':
    import DynInsSelParameters as Params
    
    from time import clock
    
    mystr = lambda number : "{:.4f}".format(number)
    
    # Choose which experiments to work on
    do_trivial = False
    do_health_groups = False
    do_age_bands = False
    do_mandate_tax = False
    do_eligibility_cutoff = False
    do_max_OOP_prem = False
    do_add_features = False
    do_del_features = False
    
    # Choose what kind of work to do
    run_experiments = False
    make_figures = False
    
    if do_trivial:
        if run_experiments:
            # Run the health groups experiments
            t_start = clock()
            MyMarket = runCounterfactual(Params.test_param_vec,BaselineSpec,[BaselineSpec],[0.,20.])
            t_end = clock()
            print('Trivial experiment took ' + mystr(t_end-t_start) + ' seconds.')
        
        if make_figures:
            # Make figures for the health groups experiments
            for specification in [BaselineSpec]:
                makeCounterfactualFigures(specification,[-1,1],[-1,1],[-1,1],[-1,1])
    
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
                makeCounterfactualFigures(specification,[-2,6],[-2,6],[-2,6],[-2,6])
                
    
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
                makeCounterfactualFigures(specification,[-2,6],[-2,6],[-2,6],[-2,6])

    
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
                makeCounterfactualFigures(specification,[-2,6],[-2,6],[-2,6],[-2,6])

                
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
                makeCounterfactualFigures(specification,[-2,10],[-2,10],[-2,10],[-2,10])
                
                
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
                makeCounterfactualFigures(specification,[-2,10],[-2,10],[-2,10],[-2,10])
                
                
    if do_add_features:
        if run_experiments:
            # Run adding one ACA feature at a time experiment
            t_start = clock()
            MyMarket = runCounterfactual(Params.test_param_vec,
                                         PreACAbaselineSpec,
                                         AddACAfeaturesSpecs,
                                         [0.,30.])
            t_end = clock()
            print('Adding ACA features counterfactual experiment took ' + mystr(t_end-t_start) + ' seconds.')
            
        if make_figures:
            # Make figures for the maximum OOP premium (FPL pct) experiments
            for specification in AddACAfeaturesSpecs:
                makeCounterfactualFigures(specification,[-2,10],[-2,10],[-2,10],[-2,10])
            makeCrossPolicyFigures('AddACAfeatures',AddACAfeaturesSpecs,[-2.,4.],[-2.,8.],[-4.,9.])
            
            
    if do_del_features:
        if run_experiments:
            # Run removing one ACA feature at a time experiment
            t_start = clock()
            MyMarket = runCounterfactual(Params.test_param_vec,
                                         PostACAbaselineSpec,
                                         DelACAfeaturesSpecs,
                                         [0.,30.])
            t_end = clock()
            print('Removing ACA features counterfactual experiment took ' + mystr(t_end-t_start) + ' seconds.')
            
        if make_figures:
            # Make figures for the maximum OOP premium (FPL pct) experiments
            for specification in DelACAfeaturesSpecs:
                makeCounterfactualFigures(specification,[-2,10],[-2,10],[-2,10],[-2,10])
            makeCrossPolicyFigures('DelACAfeatures',DelACAfeaturesSpecs,[-2.,4.],[-2.,8.],[-4.,9.])
    
    