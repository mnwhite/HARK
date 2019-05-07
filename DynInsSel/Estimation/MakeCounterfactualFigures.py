'''
This module has functions to produce figures from counterfactual experiments.
'''
import sys 
sys.path.insert(0,'../../')

import numpy as np
import matplotlib.pyplot as plt
import csv
from HARKutilities import getPercentiles, kernelRegression


def makeSinglePolicyFigures(specification,AgeHealthLim,AgeIncomeLim,AgeOfferLim,IncomeHealthLim):
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
    AgeOfferLim : [float,float]
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
    WTP_pPct = np.zeros(N,dtype=float)
    WTP_hLvl = np.zeros(N,dtype=float)
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
        WTP_pPct[i] = float(all_data[j][5])
        WTP_hLvl[i] = float(all_data[j][6])
        invalid[i] = bool(float(all_data[j][7]))
        iBefore[i] = bool(float(all_data[j][8]))
        iAfter[i] = bool(float(all_data[j][9]))  
    WTP = np.maximum(WTP_pPct,-1.)
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
            those = np.logical_and(quintiles==i, np.logical_not(offer[these]))
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
    plt.ylim(AgeOfferLim)
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
    
    
    # Make a multipanel figure plotting all WTP results on one figure
    plt.subplot(2,2,1)
    plt.plot(AgeVec,WTPmeanByAgeOffer[:,2]*100,'-k')
    plt.ylabel('All individuals')
    
    plt.subplot(2,2,2)
    plt.plot(AgeVec,WTPmeanByAgeHealth*100,'-')
    plt.ylabel('By health status')
    plt.ylim(AgeHealthLim)
    
    plt.subplot(2,2,3)
    plt.plot(AgeVec,WTPmeanByAgeOffer[:,0]*100,'-b')
    plt.plot(AgeVec,WTPmeanByAgeOffer[:,1]*100,'-r')
    plt.ylabel('By ESI status')
    plt.ylim(AgeOfferLim)
    
    plt.subplot(2,2,4)
    plt.plot(AgeVec,WTPmeanByAgeIncome*100,'-')
    plt.ylabel('By income quintile')
    plt.ylim(AgeIncomeLim)
    
    plt.suptitle('Mean WTP (as % of permanent income), ' + specification.text, y=1.02)
    plt.tight_layout()
    plt.savefig(FigsDir + specification.name + 'WTPall.pdf',bbox_inches='tight')
    plt.show()
        

def makeCrossPolicyFigures(name,specifications,AgeHealthLim,AgeIncomeLim,AgeOfferLim):
    '''
    Produces figures to graphically represent results across counterfactual
    experiments and saves them to the folder ../CounterfactualFigs.
    
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
    FigsDir = '../CounterfactualFigs/'
    
    # Initialize arrays to hold counterfactual results
    IMIinsuredRateByAge = np.zeros((P+1,T))
    IMIinsuredRateByAgeGoodHealth = np.zeros((P+1,T))
    IMIinsuredRateByAgeBadHealth = np.zeros((P+1,T))
    IMIinsuredRateByAgeHighInc = np.zeros((P+1,T))
    IMIinsuredRateByAgeMidInc = np.zeros((P+1,T))
    IMIinsuredRateByAgeLowInc = np.zeros((P+1,T))
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
        
        # Initialize data arrays
        N = len(all_data) - 1
        T = 40
        age = np.zeros(N,dtype=int)
        health = np.zeros(N,dtype=int)
        offer = np.zeros(N,dtype=int)
        mLvl = np.zeros(N,dtype=float)
        pLvl = np.zeros(N,dtype=float)
        WTP_pPct = np.zeros(N,dtype=float)
        WTP_hLvl = np.zeros(N,dtype=float)
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
            WTP_pPct[i] = float(all_data[j][5])
            WTP_hLvl[i] = float(all_data[j][6])
            invalid[i] = bool(float(all_data[j][7]))
            iBefore[i] = bool(float(all_data[j][8]))
            iAfter[i] = bool(float(all_data[j][9]))  
        WTP = np.maximum(WTP_pPct,-1.0)
        valid = np.logical_not(invalid)
        
        good_health = health >= 3
        bad_health = health < 3
        
        # Calculate IMI insured rate for this policy by age and health
        for t in range(T):
            these = np.logical_and(age == t, np.logical_not(offer))
            IMIinsuredRateByAge[p+1,t] = float(np.sum(iAfter[these]))/float(np.sum(these))
            if p==0:
                IMIinsuredRateByAge[0,t] = float(np.sum(iBefore[these]))/float(np.sum(these))
            those = np.logical_and(these, good_health)
            IMIinsuredRateByAgeGoodHealth[p+1,t] = float(np.sum(iAfter[those]))/float(np.sum(those))
            if p==0:
                IMIinsuredRateByAgeGoodHealth[0,t] = float(np.sum(iBefore[those]))/float(np.sum(those))
            those = np.logical_and(these, bad_health)
            IMIinsuredRateByAgeBadHealth[p+1,t] = float(np.sum(iAfter[those]))/float(np.sum(those))
            if p==0:
                IMIinsuredRateByAgeBadHealth[0,t] = float(np.sum(iBefore[those]))/float(np.sum(those))
        
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
            i_temp_alt = iBefore[these]
            WTP_temp = WTP[these]
            group_cuts = getPercentiles(p_temp,percentiles=[0.2,0.8])
            group = np.zeros(np.sum(these),dtype=int)
            for i in range(2):
                group[p_temp > group_cuts[i]] += 1
            IMI_temp = np.logical_not(offer[these])
            those = np.logical_and(group == 0, IMI_temp)
            IMIinsuredRateByAgeLowInc[p+1,t] = float(np.sum(i_temp[those]))/float(np.sum(those))
            if p==0:
                IMIinsuredRateByAgeLowInc[0,t] = float(np.sum(i_temp_alt[those]))/float(np.sum(those))
            those = np.logical_and(group == 1, IMI_temp)
            IMIinsuredRateByAgeMidInc[p+1,t] = float(np.sum(i_temp[those]))/float(np.sum(those))
            if p==0:
                IMIinsuredRateByAgeMidInc[0,t] = float(np.sum(i_temp_alt[those]))/float(np.sum(those))
            those = np.logical_and(group == 2, IMI_temp)
            IMIinsuredRateByAgeHighInc[p+1,t] = float(np.sum(i_temp[those]))/float(np.sum(those))
            if p==0:
                IMIinsuredRateByAgeHighInc[0,t] = float(np.sum(i_temp_alt[those]))/float(np.sum(those))
            
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
            
    
    line_styles = ['--k','-b','-g','-r','-c','-m']
    AgeVec = np.arange(25,65)
    
    plt.figure(figsize=(6.4,5.0))
    
    plt.subplot(3,2,1)
    for p in range(P+1):
        plt.plot(AgeVec,IMIinsuredRateByAge[p,:], line_styles[p])
    #plt.xlabel('Age')
    plt.ylabel('Overall')
    plt.xlim(25,65)
    plt.ylim(0.,1.)
    #plt.xticks([])
    
    plt.subplot(3,2,3)
    for p in range(P+1):
        plt.plot(AgeVec,IMIinsuredRateByAgeGoodHealth[p,:], line_styles[p])
    #plt.xlabel('Age')
    plt.ylabel('Healthy')
    plt.xlim(25,65)
    plt.ylim(0.,1.)
    #plt.xticks([])
    
    plt.subplot(3,2,5)
    for p in range(P+1):
        if p == 0:
            label = '_nolegend_'
        else:
            label = label_list[p-1]
        plt.plot(AgeVec,IMIinsuredRateByAgeBadHealth[p,:], line_styles[p], label=label)
    plt.xlabel('Age')
    plt.ylabel('Unhealthy')
    plt.xlim(25,65)
    plt.ylim(0.,1.)
    plt.legend(bbox_to_anchor=(-0.2, -0.65, 2.5, .102), loc=10,
           ncol=2, mode="expand")
    
    plt.subplot(3,2,2)
    for p in range(P+1):
        plt.plot(AgeVec,IMIinsuredRateByAgeHighInc[p,:], line_styles[p])
    #plt.xlabel('Age')
    plt.ylabel('High income')
    plt.xlim(25,65)
    plt.ylim(0.,1.)
    #plt.xticks([])
    
    plt.subplot(3,2,4)
    for p in range(P+1):
        plt.plot(AgeVec,IMIinsuredRateByAgeMidInc[p,:], line_styles[p])
    #plt.xlabel('Age')
    plt.ylabel('Mid income')
    plt.xlim(25,65)
    plt.ylim(0.,1.)
    #plt.xticks([])
    
    plt.subplot(3,2,6)
    for p in range(P+1):
        plt.plot(AgeVec,IMIinsuredRateByAgeLowInc[p,:], line_styles[p])
    plt.xlabel('Age')
    plt.ylabel('Low income')
    plt.xlim(25,65)
    plt.ylim(0.,1.)
    
    plt.suptitle('Individual market insured rate across counterfactual policies',y=1.02)
    plt.tight_layout()
    plt.savefig(FigsDir + name + 'InsuredRateIMI.pdf',bbox_inches='tight')
    plt.show()
    
    
    line_styles = ['-b','-g','-r','-c','-m']
    plt.figure(figsize=(6.4,6.6))
    
    plt.subplot(4,2,1)
    for p in range(P):
        plt.plot(AgeVec,WTPmeanByAge[p,:]*100, line_styles[p])
    plt.ylabel('Overall')
    plt.xlim(25,65)
    #plt.xticks([])
    
    plt.subplot(4,2,3)
    for p in range(P):
        plt.plot(AgeVec,WTPmeanByAgeGoodHealth[p,:]*100, line_styles[p])
    plt.ylabel('Healthy')
    plt.xlim(25,65)
    plt.ylim(AgeHealthLim[0],AgeHealthLim[1])
    #plt.xticks([])
    
    plt.subplot(4,2,5)
    for p in range(P):
        plt.plot(AgeVec,WTPmeanByAgeBadHealth[p,:]*100, line_styles[p])
    plt.ylabel('Unhealthy')
    plt.xlim(25,65)
    plt.ylim(AgeHealthLim[0],AgeHealthLim[1])
    #plt.xticks([])
    
    plt.subplot(4,2,7)
    for p in range(P):
        plt.plot(AgeVec,WTPmeanByAgeIMI[p,:]*100, line_styles[p])
    plt.xlabel('Age')
    plt.ylabel('Not offered ESI')
    plt.xlim(25,65)
    plt.ylim(AgeOfferLim[0],AgeOfferLim[1])
    plt.legend(labels=label_list,bbox_to_anchor=(-0.2, -0.65, 2.5, .102), loc=10,
           ncol=2, mode="expand")
    
    plt.subplot(4,2,2)
    for p in range(P):
        plt.plot(AgeVec,WTPmeanByAgeHighInc[p,:]*100, line_styles[p])
    plt.ylabel('High income')
    plt.xlim(25,65)
    plt.ylim(AgeIncomeLim[0],AgeIncomeLim[1])
    #plt.xticks([])
    
    plt.subplot(4,2,4)
    for p in range(P):
        plt.plot(AgeVec,WTPmeanByAgeMidInc[p,:]*100, line_styles[p])
    plt.ylabel('Mid income')
    plt.xlim(25,65)
    plt.ylim(AgeIncomeLim[0],AgeIncomeLim[1])
    #plt.xticks([])
    
    plt.subplot(4,2,6)
    for p in range(P):
        plt.plot(AgeVec,WTPmeanByAgeLowInc[p,:]*100, line_styles[p])
    plt.ylabel('Low income')
    plt.xlim(25,65)
    plt.ylim(AgeIncomeLim[0],AgeIncomeLim[1])
    #plt.xticks([])
    
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
    
    
    
def makeVaryParameterFigures(param_name,param_label,param_values,specifications,AgeLim,HealthLim,IncomeLim,OfferLim):
    '''
    Produces figures to graphically represent results across counterfactuals
    in which one policy parameter is varied; saves them to the folder ../CounterfactualFigs.
    
    Parameters
    ----------
    param_name : str
        The name of the parameter being varied, for filenames
    param_label : str
        Description of the parameter being varied, for axes label
    param_values : [float]
        List of values that the policy parameter takes on in specifications
    specifications : [ActuarialSpecification]
        Counterfactual specification whose figures are to be produced.
    AgeLim : [float,float]
        Vertical axis limits for the age WTP plot.
    HealthLim : [float,float]
        Vertical axis limits for the health WTP plot.
    IncomeLim : [float,float]
        Vertical axis limits for the income WTP plot.
    OfferLim : [float,float]
        Vertical axis limits for the offer WTP plot
        
    Returns
    -------
    None
    '''
    T = 40
    P = len(specifications)
    FigsDir = '../CounterfactualFigs/'
    
    # Define age group (min,max)
    age_groups = [[0,9],[10,19],[20,29],[30,39]]
    
    # Initialize arrays to hold counterfactual results
    IMIinsuredRateTotal = np.zeros((P))
    IMIinsuredRateByAge = np.zeros((P,len(age_groups)))
    IMIinsuredRateByHealth = np.zeros((P,5))
    IMIinsuredRateByIncome = np.zeros((P,5))
    WTPmeanByAge = np.zeros((P,len(age_groups)))
    WTPmeanByHealth = np.zeros((P,5))
    WTPmeanByIncome = np.zeros((P,5))
    WTPmeanByOffer = np.zeros((P,2))
    label_list = []
    
    for p in range(P):
        specification = specifications[p]
        print('Processing the specification labeled ' + specification.text + '...')
        label_list.append(specification.text)
        
        with open('../Results/' + specification.name + 'Data.txt','r') as f:
            my_reader = csv.reader(f, delimiter = '\t')
            all_data = list(my_reader)
        
        # Initialize data arrays
        N = len(all_data) - 1
        T = 40
        age = np.zeros(N,dtype=int)
        health = np.zeros(N,dtype=int)
        offer = np.zeros(N,dtype=int)
        mLvl = np.zeros(N,dtype=float)
        pLvl = np.zeros(N,dtype=float)
        WTP_pPct = np.zeros(N,dtype=float)
        WTP_hLvl = np.zeros(N,dtype=float)
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
            WTP_pPct[i] = float(all_data[j][5])
            WTP_hLvl[i] = float(all_data[j][6])
            invalid[i] = bool(float(all_data[j][7]))
            iBefore[i] = bool(float(all_data[j][8]))
            iAfter[i] = bool(float(all_data[j][9]))  
        WTP = WTP_pPct
        valid = np.logical_not(invalid)
        IMI = np.logical_not(offer)
        
        # Loop through each age and label observations by income quintile
        inc_quint = -np.ones_like(health)
        for t in range(T):
            these = age == t
            p_temp = pLvl[these]
            quintile_cuts = getPercentiles(p_temp,percentiles=[0.2,0.4,0.6,0.8])
            quint_temp = np.zeros(np.sum(these),dtype=int)
            for i in range(4):
                quint_temp[p_temp > quintile_cuts[i]] += 1
            inc_quint[these] = quint_temp
            
        # Calculate overall IMI insured rate
        these = np.logical_not(offer)
        IMIinsuredRateTotal[p] = float(np.sum(iAfter[these]))/float(np.sum(these))
        
        # Loop over age groups and calculate IMI insured rate within each
        for g in range(len(age_groups)):
            these = np.logical_and(age >= age_groups[g][0], age <= age_groups[g][1])
            these = np.logical_and(these,IMI)
            IMIinsuredRateByAge[p,g] = float(np.sum(iAfter[these]))/float(np.sum(these))
            
        # Loop over health states and calculate IMI insured rate within each
        for h in range(5):
            these = health == h
            these = np.logical_and(these,IMI)
            IMIinsuredRateByHealth[p,h] = float(np.sum(iAfter[these]))/float(np.sum(these))
            
        # Loop over income quintiles and calculate IMI insured rate within each
        for i in range(5):
            these = inc_quint == i
            these = np.logical_and(these,IMI)
            IMIinsuredRateByIncome[p,i] = float(np.sum(iAfter[these]))/float(np.sum(these))
            
        # Loop over age groups and calculate WTP mean within each
        for g in range(len(age_groups)):
            these = np.logical_and(age >= age_groups[g][0], age <= age_groups[g][1])
            these = np.logical_and(these,valid)
            WTPmeanByAge[p,g] = np.mean(WTP[these])
            
        # Loop over health states and calculate WTP mean within each
        for h in range(5):
            these = health == h
            these = np.logical_and(these,valid)
            WTPmeanByHealth[p,h] = np.mean(WTP[these])
            
        # Loop over income quintiles and calculate WTP mean within each
        for i in range(5):
            these = inc_quint == i
            these = np.logical_and(these,valid)
            WTPmeanByIncome[p,i] = np.mean(WTP[these])
            
        # Calculate WTP mean by offer status
        these = np.logical_and(IMI,valid)
        WTPmeanByOffer[p,0] = np.mean(WTP[these])
        these = np.logical_and(offer,valid)
        WTPmeanByOffer[p,1] = np.mean(WTP[these])
        
        
    # Make a four panel plot of insured rate while varying the policy parameter
    line_styles = ['-b','-g','-r','-c','-m']
    ParamVec = np.array(param_values)
    
    plt.subplot(2,2,1)
    plt.plot(ParamVec,IMIinsuredRateTotal[:], '-k')
    plt.ylabel('Overall')
    plt.ylim(0.,1.)
    
    plt.subplot(2,2,2)
    for g in range(len(age_groups)):
        plt.plot(ParamVec,IMIinsuredRateByAge[:,g], line_styles[g])
    plt.ylabel('By age group')
    plt.ylim(0.,1.)
    
    plt.subplot(2,2,3)
    for h in range(5):
        plt.plot(ParamVec,IMIinsuredRateByHealth[:,h], line_styles[h])
    plt.ylabel('By health status')
    plt.ylim(0.,1.)
    plt.xlabel(param_label)
    
    plt.subplot(2,2,4)
    for i in range(5):
        plt.plot(ParamVec,IMIinsuredRateByIncome[:,i], line_styles[i])
    plt.ylabel('By income quintile')
    plt.ylim(0.,1.)
    plt.xlabel(param_label)
    
    plt.suptitle('Individual market insured rate by ' + param_label,y=1.02)
    plt.tight_layout()
    plt.savefig(FigsDir + 'Vary' + param_name + 'InsuredRateIMI.pdf',bbox_inches='tight')
    plt.show()
    
    
    
    # Make a four panel plot of WTP mean while varying the policy parameter
    line_styles = ['-b','-g','-r','-c','-m']
    ParamVec = np.array(param_values)
    
    plt.subplot(2,2,1)
    plt.plot(ParamVec,WTPmeanByOffer[:,0]*100, '-r')
    plt.plot(ParamVec,WTPmeanByOffer[:,1]*100, '-b')
    plt.ylabel('By ESI status')
    plt.ylim(OfferLim[0],OfferLim[1])
    
    plt.subplot(2,2,2)
    for g in range(len(age_groups)):
        plt.plot(ParamVec,WTPmeanByAge[:,g]*100, line_styles[g])
    plt.ylabel('By age group')
    plt.ylim(AgeLim[0],AgeLim[1])
    
    plt.subplot(2,2,3)
    for h in range(5):
        plt.plot(ParamVec,WTPmeanByHealth[:,h]*100, line_styles[h])
    plt.ylabel('By health status')
    plt.ylim(HealthLim[0],HealthLim[1])
    plt.xlabel(param_label)
    
    plt.subplot(2,2,4)
    for i in range(5):
        plt.plot(ParamVec,WTPmeanByIncome[:,i]*100, line_styles[i])
    plt.ylabel('By income quintile')
    plt.ylim(IncomeLim[0],IncomeLim[1])
    plt.xlabel(param_label)
    
    plt.suptitle('Mean WTP (as % of permanent income) by ' + param_label,y=1.02)
    plt.tight_layout()
    plt.savefig(FigsDir + 'Vary' + param_name + 'WTPmean.pdf',bbox_inches='tight')
    plt.show()
            

def makePremiumFigure(name,specifications,post_bool):
    '''
    Produces figure to graphically represent premiums across counterfactual
    experiments and saves them to the folder ../CounterfactualFigs.
    
    Parameters
    ----------
    name : str
        Filename prefix for this set of specifications
    specifications : [ActuarialSpecification]
        Counterfactual specification whose premiums are to be plotted.
    post_bool : bool
        If True, baseline for these counterfactuals is the post-ACA world.
        If False, baseline is the pre-ACA world
        
    Returns
    -------
    None
    '''
    T = 40
    P = len(specifications)
    FigsDir = '../CounterfactualFigs/'
    
    PremiumArray = np.zeros((T,P))
    label_list = []
    
    if post_bool:
        prefix = 'Post'
    else:
        prefix = 'Pre'
        
    with open('../Results/' + prefix + 'ACAbaselinePremiums.txt','r') as f:
            my_reader = csv.reader(f, delimiter = '\t')
            all_data = list(my_reader)
    BaselinePremiums = np.array(all_data[0])
    label_list.append(prefix + ' ACA baseline')
        
    for p in range(P):
        specification = specifications[p]
        label_list.append(specification.text)
        
        with open('../Results/' + specification.name + 'Premiums.txt','r') as f:
            my_reader = csv.reader(f, delimiter = '\t')
            all_data = list(my_reader)
            
        PremiumArray[:,p] = np.array(all_data[0])
    
    line_styles = ['-b','-g','-r','-c','-m']
    AgeVec = np.arange(25,65)
    
    plt.plot(AgeVec,BaselinePremiums,'-k')
    for p in range(P):
        plt.plot(AgeVec,PremiumArray[:,p]*10,line_styles[p])
    plt.xlabel('Age')
    plt.ylabel('Annual premium (thousands of USD)')
    plt.title('Individual market premiums across policies')
    plt.legend(labels=label_list)
    plt.savefig(FigsDir + name + 'Premiums.pdf')
    plt.show()

    