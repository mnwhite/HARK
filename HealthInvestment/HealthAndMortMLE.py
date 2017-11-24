'''
This module runs two simple maximum likelihood estimations to get "reduced form"
estimates of the mortality process and health transition process.
'''

import sys
import os
import csv
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
sys.path.insert(0,'../')
sys.path.insert(0,'./Data/')

# Load the estimation data into memory
infile = open('./Data/EstimationData.txt','r') 
my_reader = csv.reader(infile,delimiter='\t')
all_data = list(my_reader)
infile.close()
obs = len(all_data)-1

# Initialize numpy arrays for the data
typenum_data = np.zeros(obs,dtype=int)
sex_data = np.zeros(obs,dtype=int)
inc_quint_data = np.zeros(obs,dtype=int)
cohort_data = np.zeros(obs,dtype=int)
wealth_quint_data = np.zeros(obs,dtype=int)
health_tert_data = np.zeros(obs,dtype=int)
first_ob_data = np.zeros(obs,dtype=int)
w0_data = np.zeros(obs) # 1996 wealth
w1_data = np.zeros(obs) # 1998 wealth
w2_data = np.zeros(obs) # 2000 wealth
w3_data = np.zeros(obs) # 2002 wealth
w4_data = np.zeros(obs) # 2004 wealth
w5_data = np.zeros(obs) # 2006 wealth
w6_data = np.zeros(obs) # 2008 wealth
w7_data = np.zeros(obs) # 2010 wealth
h0_data = np.zeros(obs) # 1996 health
h1_data = np.zeros(obs) # 1998 health
h2_data = np.zeros(obs) # 2000 health
h3_data = np.zeros(obs) # 2002 health
h4_data = np.zeros(obs) # 2004 health
h5_data = np.zeros(obs) # 2006 health
h6_data = np.zeros(obs) # 2008 health
h7_data = np.zeros(obs) # 2010 health
m0_data = np.zeros(obs) # 1996 OOPmed
m1_data = np.zeros(obs) # 1998 OOPmed
m2_data = np.zeros(obs) # 2000 OOPmed
m3_data = np.zeros(obs) # 2002 OOPmed
m4_data = np.zeros(obs) # 2004 OOPmed
m5_data = np.zeros(obs) # 2006 OOPmed
m6_data = np.zeros(obs) # 2008 OOPmed
m7_data = np.zeros(obs) # 2010 OOPmed

# Unpack data into numpy arrays
for i in range(obs):
    j = i+1
    typenum_data[i] = int(all_data[j][0])
    sex_data[i] = int(all_data[j][1])
    inc_quint_data[i] = int(all_data[j][2])
    cohort_data[i] = int(all_data[j][3])
    wealth_quint_data[i] = int(all_data[j][4])
    health_tert_data[i] = int(all_data[j][5])
    first_ob_data[i] = int(all_data[j][6])
    w0_data[i] = float(all_data[j][7])
    w1_data[i] = float(all_data[j][8])
    w2_data[i] = float(all_data[j][9])
    w3_data[i] = float(all_data[j][10])
    w4_data[i] = float(all_data[j][11])
    w5_data[i] = float(all_data[j][12])
    w6_data[i] = float(all_data[j][13])
    w7_data[i] = float(all_data[j][14])
    h0_data[i] = float(all_data[j][15])
    h1_data[i] = float(all_data[j][16])
    h2_data[i] = float(all_data[j][17])
    h3_data[i] = float(all_data[j][18])
    h4_data[i] = float(all_data[j][19])
    h5_data[i] = float(all_data[j][20])
    h6_data[i] = float(all_data[j][21])
    h7_data[i] = float(all_data[j][22])
    m0_data[i] = float(all_data[j][23])
    m1_data[i] = float(all_data[j][24])
    m2_data[i] = float(all_data[j][25])
    m3_data[i] = float(all_data[j][26])
    m4_data[i] = float(all_data[j][27])
    m5_data[i] = float(all_data[j][28])
    m6_data[i] = float(all_data[j][29])
    m7_data[i] = float(all_data[j][30])
    
# Slightly process the data, relabeling -1 as nan
w0_data[w0_data==-1.] = np.nan
w1_data[w1_data==-1.] = np.nan
w2_data[w2_data==-1.] = np.nan
w3_data[w3_data==-1.] = np.nan
w4_data[w4_data==-1.] = np.nan
w5_data[w5_data==-1.] = np.nan
w6_data[w6_data==-1.] = np.nan
w7_data[w7_data==-1.] = np.nan
h0_data[h0_data==-1.] = np.nan
h1_data[h1_data==-1.] = np.nan
h2_data[h2_data==-1.] = np.nan
h3_data[h3_data==-1.] = np.nan
h4_data[h4_data==-1.] = np.nan
h5_data[h5_data==-1.] = np.nan
h6_data[h6_data==-1.] = np.nan
h7_data[h7_data==-1.] = np.nan
m0_data[m0_data==-1.] = np.nan
m1_data[m1_data==-1.] = np.nan
m2_data[m2_data==-1.] = np.nan
m3_data[m3_data==-1.] = np.nan
m4_data[m4_data==-1.] = np.nan
m5_data[m5_data==-1.] = np.nan
m6_data[m6_data==-1.] = np.nan
m7_data[m7_data==-1.] = np.nan
inc_quint_data[inc_quint_data==0] = 5

# Combine the data by year
w_data = np.vstack((w0_data,w1_data,w2_data,w3_data,w4_data,w5_data,w6_data,w7_data))
h_data = np.vstack((h0_data,h1_data,h2_data,h3_data,h4_data,h5_data,h6_data,h7_data))
m_data = np.vstack((m0_data,m1_data,m2_data,m3_data,m4_data,m5_data,m6_data,m7_data))*10000
t0 = (first_ob_data-1996)/2
idx = np.arange(obs)
N = np.sum(cohort_data <= 15)

# Make an array of age and sex
age_data = 11 + np.tile(np.reshape(np.arange(7),(7,1)),(1,obs)) - np.tile(np.reshape(cohort_data,(1,obs)),(7,1))
sex_big_data = np.tile(np.reshape(sex_data,(1,obs)),(7,1))

# Choose the data to be used in the health transition and mortality MLEs
EstimationCohorts = np.zeros_like(h_data[:7,:],dtype=bool)
EstimationCohorts[:,:N] = True
AliveAtT = h_data[:7,:] > 0.
ObservedAtTp1 = np.logical_not(np.isnan(h_data[1:,:]))
AliveAtTp1 = h_data[1:,:] > 0.
UseForHealthTrans = np.logical_and(EstimationCohorts,np.logical_and(AliveAtT,AliveAtTp1))
UseForMortality = np.logical_and(EstimationCohorts,np.logical_and(AliveAtT,ObservedAtTp1))

# Select the data for the health transition MLE
HealthNow_trans = h_data[:7,:][UseForHealthTrans]
Age_trans = age_data[UseForHealthTrans]
HealthSqNow_trans = HealthNow_trans**2
AgeSq_trans = Age_trans**2
Sex_trans = sex_big_data[UseForHealthTrans]
HealthNext_trans = h_data[1:,:][UseForHealthTrans]
h_temp = HealthNext_trans - 0.000001
HealthNextBot = np.floor(h_temp*100)/100
HealthNextTop = np.ceil(h_temp*100)/100
HealthNextBot[HealthNextBot == 0.] = -np.inf
HealthNextTop[HealthNextTop == 1.] = np.inf

# Select the data for the mortality MLE (probit)
HealthNow_mort = h_data[:7,:][UseForMortality]
Age_mort = age_data[UseForMortality]
HealthSqNow_mort = HealthNow_mort**2
AgeSq_mort = Age_mort**2
Sex_mort = sex_big_data[UseForMortality]
Died_mort = h_data[1:,:][UseForMortality] == 0.
Lived_mort = np.logical_not(Died_mort)


    

# Define the log likelihood function for the mortality probit
def MortalityLL(params):
    '''
    Mortality probit log likelihood function.  Takes a size 6 vector of parameters
    as an input and returns the log likelihood of these parameters given data.
    Parameter order: constant, sex, age, age_sq, health, health_sq.
    '''
    # Unpack the parameters
    Mortality0 = params[0]
    MortalitySex = params[1]
    MortalityAge = params[2]
    MortalityAgeSq = params[3]
    MortalityHealth = params[4]
    MortalityHealthSq = params[5]
    
    # Calculate the Z value of the mortality process
    Z = Mortality0 + Sex_mort*MortalitySex + Age_mort*MortalityAge + AgeSq_mort*MortalityAgeSq + HealthNow_mort*MortalityHealth + HealthSqNow_mort*MortalityHealthSq
    
    # Calculate the probability of observed mortality event given the Z score
    MortProb = np.zeros_like(HealthNow_mort)
    MortProb[Lived_mort] = norm.sf(Z[Lived_mort])
    MortProb[Died_mort] = norm.cdf(Z[Died_mort])
    
    # Calculate and return the log likelihood of the mortality data
    LogMortProb = np.log(MortProb)
    LogLikelihood = np.sum(LogMortProb)
    #print(LogLikelihood)
    return LogLikelihood
    
def NegMortLL(params):
    return -MortalityLL(params)
    
    
# Define the log likelihood function for the health transition MLE
def HealthTransLL(params):
    '''
    Health transition log likelihood function.  Takes a size 8 vector of parameters
    as an input and returns the log likelihood of these parameters given data.
    Parameter order: constant, sex, age, age_sq, health, health_sq, stdev0, stdev_health.
    '''
    # Unpack the parameters
    HealthNext0 = params[0]
    HealthNextSex = params[1]
    HealthNextAge = params[2]
    HealthNextAgeSq = params[3]
    HealthNextHealth = params[4]
    HealthNextHealthSq = params[5]
    HealthShkStd0 = params[6]
    HealthShkStd1 = params[7]
    
    # Calculate expected health next for each observation
    ExpHealthNext = HealthNext0 + Sex_trans*HealthNextSex + Age_trans*HealthNextAge + AgeSq_trans*HealthNextAgeSq + HealthNow_trans*HealthNextHealth + HealthSqNow_trans*HealthNextHealthSq
    
    # Calculate health shock stdev
    ShockStd = HealthShkStd0 + HealthShkStd1*HealthNow_trans
    
    # Calculate the likelihood of each health transition
    Z_bot = (HealthNextBot - ExpHealthNext)/ShockStd
    Z_top = (HealthNextTop - ExpHealthNext)/ShockStd
    TransProb = norm.cdf(Z_top) - norm.cdf(Z_bot)
    
    # Calculate and return the probability of observed health transition
    LogTransProb = np.log(TransProb)
    LogLikelihood = np.sum(LogTransProb)
    #print(LogLikelihood)
    return LogLikelihood
    
def NegTransLL(params):
    return -HealthTransLL(params)
    
    
if __name__ == '__main__':
    import HealthInvParams as Params
    from time import clock
    from copy import copy
    from HARKestimation import minimizeNelderMead
    
    estimate_mortality = True
    estimate_health_trans = False
    
    mort_test_params = Params.test_param_vec[27:33]
    trans_test_params = Params.test_param_vec[16:24]
    
#    # Test model identification by perturbing one parameter at a time
#    param_i = 0
#    param_min = 0.0
#    param_max = 0.1
#    N = 100
#    perturb_vec = np.linspace(param_min,param_max,num=N)
#    fit_vec = np.zeros(N) + np.nan
#    for j in range(N):
#        params = copy(trans_test_params)
#        params[param_i] = perturb_vec[j]
#        fit_vec[j] = NegTransLL(params)
#        
#    plt.plot(perturb_vec,fit_vec)
#    plt.xlabel(Params.param_names[param_i+16])
#    plt.ylabel('Sum of squared moment differences')
#    plt.show()


    if estimate_mortality:
        t0 = clock()
        MortalityLL(mort_test_params)
        t1 = clock()
        print('One mortality LL evaluation took ' + str(t1-t0) + ' seconds.')
        
        # Estimate some (or all) of the model parameters for the mortality MLE
        which_indices = np.array([0,1,2,3,4,5])
        which_bool = np.zeros(6,dtype=bool)
        which_bool[which_indices] = True
        estimated_params = minimizeNelderMead(NegMortLL,mort_test_params,verbose=True,which_vars=which_bool)
        for i in which_indices.tolist():
            print(Params.param_names[i+27] + ' = ' + str(estimated_params[i]))
            
        # Unpack the parameters
        Mortality0 = estimated_params[0]
        MortalitySex = estimated_params[1]
        MortalityAge = estimated_params[2]
        MortalityAgeSq = estimated_params[3]
        MortalityHealth = estimated_params[4]
        MortalityHealthSq = estimated_params[5]
        
        # Calculate the Z value of the mortality process
        Z = Mortality0 + Sex_mort*MortalitySex + Age_mort*MortalityAge + AgeSq_mort*MortalityAgeSq + HealthNow_mort*MortalityHealth + HealthSqNow_mort*MortalityHealthSq
    
        # Calculate the predicted probability of observed mortality event given the Z score
        PredictedMortProb = norm.cdf(Z)
                    
        death_prob_data = np.zeros(15)
        death_prob_model = np.zeros(15)
        for a in range(15):
            these = Age_mort == a
            data_deaths = float(np.sum(Died_mort[these]))
            data_total = float(np.sum(these))
            death_prob_data[a] = data_deaths/data_total
            death_prob_model[a] = np.mean(PredictedMortProb[these])
        plt.plot(death_prob_data,'.k')
        plt.plot(death_prob_model,'-')
        plt.show()
            
            
    if estimate_health_trans:
        t0 = clock()
        HealthTransLL(trans_test_params)
        t1 = clock()
        print('One health transition LL evaluation took ' + str(t1-t0) + ' seconds.')
        
        # Estimate some (or all) of the model parameters for the mortality MLE
        which_indices = np.array([0,1,2,3,4,5,6,7])
        which_bool = np.zeros(8,dtype=bool)
        which_bool[which_indices] = True
        estimated_params = minimizeNelderMead(NegTransLL,trans_test_params,verbose=True,which_vars=which_bool)
        for i in which_indices.tolist():
            print(Params.param_names[i+16] + ' = ' + str(estimated_params[i]))
    
    

