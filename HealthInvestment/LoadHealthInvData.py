'''
This module imports data for the health investment estimation from text files
exported from Stata.
'''

import sys
import os
import csv
import numpy as np
import scipy as sp
import statsmodels.api as sm
sys.path.insert(0,'../')
sys.path.insert(0,'./Data/')
from HARKutilities import getPercentiles

# Choose how many times to bootstrap the data to calculate standard errors for data moments.
# If this is zero, the module will try to read the CSV file ./Data/MomentWeights.txt to load
# the weighting matrix.  If this is non-zero, the moment weights will be created and saved.
data_bootstrap_count = 0

# Choose whether to use cohorts in simulation (True) or aggregate into only 10 types (False)
use_cohorts = False

# Choose which moments will actually be used
moment_dummies = np.array([
        False, # OOPbyAge
        False, # StDevOOPbyAge
        False, # MortByAge
        False, # StDevDeltaHealthByAge
        False, # StDevOOPbyHealthAge
        False, # StDevDeltaHealthByHealthAge
        True,  # HealthBySexHealthAge
        False, # OOPbySexHealthAge
        False, # MortBySexHealthAge
        False, # WealthByIncAge
        True,  # HealthByIncAge
        False, # OOPbyIncAge
        False, # WealthByIncWealthAge
        False, # HealthByIncWealthAge
        False, # OOPbyIncWealthAge
        False, # AvgHealthResidualByIncWealth
        False, # AvgOOPResidualByIncWealth
        False, # MortByHealthAge
        True,  # HealthByHealthAge
        ])

# Make a random number generator for the data bootstrap
RNG = np.random.RandomState(seed=112883)

# Load the estimation data into memory
infile = open('./Data/EstimationData.txt','r') 
my_reader = csv.reader(infile,delimiter='\t')
all_data = list(my_reader)
infile.close()
obs = len(all_data)-1

def arsinh(x):
    #return np.log(x + np.sqrt(x**2+1.))
    return x

# Initialize numpy arrays for the data
typenum_data_orig = np.zeros(obs,dtype=int)
sex_data_orig = np.zeros(obs,dtype=int)
inc_quint_data_orig = np.zeros(obs,dtype=int)
cohort_data_orig = np.zeros(obs,dtype=int)
wealth_quint_data_orig = np.zeros(obs,dtype=int)
health_tert_data_orig = np.zeros(obs,dtype=int)
first_ob_data_orig = np.zeros(obs,dtype=int)
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
    typenum_data_orig[i] = int(all_data[j][0])
    sex_data_orig[i] = int(all_data[j][1])
    inc_quint_data_orig[i] = int(all_data[j][2])
    cohort_data_orig[i] = int(all_data[j][3])
    wealth_quint_data_orig[i] = int(all_data[j][4])
    health_tert_data_orig[i] = int(all_data[j][5])
    first_ob_data_orig[i] = int(all_data[j][6])
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
inc_quint_data_orig[inc_quint_data_orig==0] = 5

# Combine the data by year
w_data_orig = np.vstack((w0_data,w1_data,w2_data,w3_data,w4_data,w5_data,w6_data,w7_data))
h_data_orig = np.vstack((h0_data,h1_data,h2_data,h3_data,h4_data,h5_data,h6_data,h7_data))
m_data_orig = np.vstack((m0_data,m1_data,m2_data,m3_data,m4_data,m5_data,m6_data,m7_data))
idx = np.arange(obs)

# Load the income profiles into memory
infile = open('./Data/IncProfilesNew.txt','r') 
my_reader = csv.reader(infile,delimiter='\t')
all_data = list(my_reader)
infile.close()
TypeCount = len(all_data) - 1

# Extract income and maximum wealth data into arrays
IncomeArray = np.zeros((TypeCount,25))
MaxWealth = np.zeros(TypeCount)
for i in range(TypeCount):
    j = i+1
    MaxWealth[i] = float(all_data[j][1])
    for t in range(21):
        IncomeArray[i,t] = float(all_data[j][t+2])
IncomeArray[:,21:] = np.tile(np.reshape(IncomeArray[:,20],(TypeCount,1)),(1,4))

# Make a "small" array of income profiles by sex and income quintile
IncomeArraySmall = np.zeros((10,25))
MaxWealthSmall = np.zeros(10)
for j in range(10):
    these = j + np.arange(0,150,10)
    IncomeArraySmall[j,:] = np.mean(IncomeArray[these,:],axis=0)
    MaxWealthSmall[j] = np.max(MaxWealth[these])

# Initialize an array of bootstrapped data moments
moment_count = 1970
BootstrappedMoments = np.zeros((data_bootstrap_count,moment_count)) + np.nan
BootstrapValidBool = np.zeros(data_bootstrap_count,dtype=bool)
BootstrappedResiduals = np.zeros((data_bootstrap_count,26)) + np.nan

# Loop over bootstrap runs; if this is the very last pass, use the real data instead
for b in range(data_bootstrap_count+1):
    # If this is not the last pass, then resample the data within each cohort
    if b < data_bootstrap_count:
        # Initialize arrays to hold this set of resampled data
        w_data = np.zeros_like(w_data_orig)
        h_data = np.zeros_like(h_data_orig)
        m_data = np.zeros_like(m_data_orig)
        first_ob_data = np.zeros_like(first_ob_data_orig)
        typenum_data = np.zeros_like(typenum_data_orig)
        cohort_data = np.zeros_like(cohort_data_orig)
        inc_quint_data = np.zeros_like(inc_quint_data_orig)
        wealth_quint_data = np.zeros_like(wealth_quint_data_orig)
        health_tert_data = np.zeros_like(health_tert_data_orig)
        sex_data = np.zeros_like(sex_data_orig)
        
        # Resample the data within each cohort
        for c in range(1,19):
            # Get indices for resampling
            THESE = cohort_data_orig == c
            cohort_idx = np.where(THESE)[0]
            cohort_N = np.sum(THESE)
            draws = RNG.randint(cohort_N,size=cohort_N)
            new_idx = cohort_idx[draws]
            
            # Fill in parts of the data arrays with resampled observations
            w_data[:,THESE] = w_data_orig[:,new_idx]
            h_data[:,THESE] = h_data_orig[:,new_idx]
            m_data[:,THESE] = m_data_orig[:,new_idx]
            first_ob_data[THESE] = first_ob_data_orig[new_idx]
            typenum_data[THESE] = typenum_data_orig[new_idx]
            cohort_data[THESE] = cohort_data_orig[new_idx]
            inc_quint_data[THESE] = inc_quint_data_orig[new_idx]
            wealth_quint_data[THESE] = wealth_quint_data_orig[new_idx]
            health_tert_data[THESE] = health_tert_data_orig[new_idx]
            sex_data[THESE] = sex_data_orig[new_idx]            
        
    else: # If this is the last pass, just load the data itself
        w_data = w_data_orig
        h_data = h_data_orig
        m_data = m_data_orig
        first_ob_data = first_ob_data_orig
        typenum_data = typenum_data_orig
        cohort_data = cohort_data_orig
        inc_quint_data = inc_quint_data_orig
        wealth_quint_data = wealth_quint_data_orig
        health_tert_data = health_tert_data_orig
        sex_data = sex_data_orig

    t0 = (first_ob_data-1996)/2
    InitBoolArray = np.zeros((8,obs),dtype=bool)
    InitBoolArray[t0,idx] = True
    
    # Get initial data states
    w_init = w_data[t0,idx]
    h_init = h_data[t0,idx]
    
    # Make boolean arrays of types (with and without cohorts)
    N = 180
    TypeBoolArray = np.zeros((N,obs),dtype=bool)
    for n in range(N):
        TypeBoolArray[n,:] = typenum_data == n+1
    typenum_temp = np.mod(typenum_data - 1,10)
    TypeBoolArraySmall = np.zeros((10,obs),dtype=bool)
    TypeBoolArrayCounterfactual = np.zeros((10,obs),dtype=bool)
    data_exists_2010 = np.logical_and(np.logical_not(np.isnan(w7_data)), np.logical_not(np.isnan(h7_data)))
    alive_2010 = h7_data > 0.
    counterfactual_valid = np.logical_and(data_exists_2010, alive_2010)
    age_in_2010 = 18 - cohort_data
    for n in range(10):
        TypeBoolArraySmall[n,:] = np.logical_and(typenum_temp == n, typenum_data <= 150)
        counterfactual_cohorts = np.logical_and(typenum_temp == n, typenum_data > 150)
        TypeBoolArrayCounterfactual[n,:] = np.logical_and(counterfactual_cohorts, counterfactual_valid)
    
    # Make a boolean array indicating when each agent is "born" in the data
    born_t = np.maximum(t0 + 11 - cohort_data,0)
    BornBoolArray = np.zeros((25,obs),dtype=bool)
    BornBoolArray[born_t,idx] = True
    
    # Make a boolean array indicating which periods of life *could* be observed in the data for each agent
    DataStartByCohort = np.concatenate([np.arange(10,0,-1), np.zeros(8,dtype=int)])
    DataEndByCohort = np.arange(17,-1,-1)
    InDataSpanArray = np.zeros((25,obs),dtype=bool)
    for j in range(18):
        c = j+1 # cohort number
        these = cohort_data == c
        N = np.sum(these)
        temp_bool = np.zeros((25,1),dtype=bool)
        temp_bool[DataStartByCohort[j]:DataEndByCohort[j]] = True
        temp_bool_rep = np.tile(temp_bool,(1,N))
        InDataSpanArray[:,these] = temp_bool_rep
    
    # Make an array of "ages" for each observation (plus boolean version)
    turn65_t = cohort_data - 11
    age_data = np.tile(np.reshape(np.arange(8),(8,1)),(1,obs)) - np.tile(np.reshape(turn65_t,(1,obs)),(8,1))
    AgeBoolArray = np.zeros((8,obs,15),dtype=bool)
    for j in range(15):
        right_age = age_data == j+1
        AgeBoolArray[:,:,j] = right_age
    NotTooOld = (age_data < 14)[:7,:] # unused?
        
    # Make a boolean array of usable observations (for non-mortality moments)
    BelowCohort16 = np.tile(np.reshape(cohort_data,(1,obs)),(8,1)) < 16
    Alive = h_data > 0.
    NotInit = np.logical_not(InitBoolArray)
    Useable = np.logical_and(BelowCohort16,np.logical_and(Alive,NotInit))
    
    # Calculate health quintiles at data entry by age (and overwrite old health tertile info)
    health_tertile_cuts = np.zeros((2,15))
    health_quintile_cuts = np.zeros((4,15))
    health_quint_data = np.zeros_like(health_tert_data,dtype=int)
    for a in range(15):
        these = np.logical_and(Alive,AgeBoolArray[:,:,a])
        health_temp = h_data[these]
        health_quintile_cuts[:,a] = getPercentiles(health_temp,percentiles=[0.2,0.4,0.6,0.8])
        health_tertile_cuts[:,a] = getPercentiles(health_temp,percentiles=[0.333,0.666])
        those = BornBoolArray[a,:]
        h_init_temp = h_init[those]
        health_quint_data[those] = np.searchsorted(health_quintile_cuts[:,a],h_init_temp) + 1
        health_tert_data[those] = np.searchsorted(health_tertile_cuts[:,a],h_init_temp) + 1
    
    # Make data objects for the health production pre-estimation
    UseableAlt = np.logical_and(BelowCohort16,Alive)
    inc_quint_data_rep = np.tile(np.reshape(inc_quint_data,(1,obs)),(8,1))
    wealth_quint_data_rep = np.tile(np.reshape(wealth_quint_data,(1,obs)),(8,1))
    sex_data_rep = np.tile(np.reshape(sex_data,(1,obs)),(8,1))
    WealthArraysBySexIncAge = []
    HealthArraysBySexIncAge = []
    WealthQuintArraysBySexIncAge = []
    for s in range(2):
        for i in range(5):
            j = i+1
            TempHealth = []
            TempWealth = []
            TempQuint = []
            these = np.logical_and(UseableAlt,np.logical_and(inc_quint_data_rep==j,sex_data_rep==s))
            for a in range(15):
                those = np.logical_and(these,age_data==a)
                TempHealth.append(h_data[those])
                TempWealth.append(w_data[those])
                TempQuint.append(wealth_quint_data_rep[those])
            WealthArraysBySexIncAge.append(TempWealth)
            HealthArraysBySexIncAge.append(TempHealth)
            WealthQuintArraysBySexIncAge.append(TempQuint)
              
    # Make a boolean array of usable observations (for mortality moments)
    AliveLastPeriod = np.zeros_like(h_data,dtype=bool)
    AliveLastPeriod[1:,:] = Alive[:-1,:]
    ObsThisPeriod = np.logical_not(np.isnan(h_data))
    DeadThisPeriod = h_data == 0.
    MortUseable = np.logical_and(AliveLastPeriod,ObsThisPeriod)
    JustDied = np.logical_and(AliveLastPeriod,DeadThisPeriod)
    
    # Make a boolean array of usable observations (for stdev health delta moments)
    HealthDeltaUseable = np.zeros_like(h_data,dtype=bool)
    HealthDeltaUseable[1:,:] = np.logical_and(AliveLastPeriod[1:,:],Alive[1:,:])
    HealthDelta = np.zeros((8,obs))
    HealthDelta[1:,:] = h_data[1:,:] - h_data[:-1,:]
    
    # Make flattened arrays of age, health, sex, and income and wealth quintiles (for health regression)
    Temp = np.logical_and(HealthDeltaUseable[1:,:],BelowCohort16[1:,:])
    HealthFlat = h_data[0:7,:][Temp]
    hNextFlat = h_data[1:,:][Temp]
    HealthSqFlat = HealthFlat**2
    AgeFlat = age_data[0:7,:][Temp]
    AgeSqFlat = AgeFlat**2
    SexFlat = np.tile(np.reshape(sex_data,(1,obs)),(7,1))[Temp]
    IQflat = np.tile(np.reshape(inc_quint_data,(1,obs)),(7,1))[Temp]
    WQflat = np.tile(np.reshape(wealth_quint_data,(1,obs)),(7,1))[Temp]
    
    # Make flat arrays with inc-wealth indicators (for health regression)
    IWQboolFlat = np.zeros((25,hNextFlat.size))
    k = 0
    for i in range(5):
        right_inc = IQflat == i+1
        for j in range(5):
            right_wealth = WQflat == j+1
            these = np.logical_and(right_inc,right_wealth)
            IWQboolFlat[k,these] = 1.0
            k += 1
    
    # Run a basic regression to predict health next period
    regressors = np.transpose(np.vstack([np.ones_like(HealthFlat),SexFlat,HealthFlat,HealthSqFlat,AgeFlat,AgeSqFlat,IWQboolFlat[3:,:]]))
    health_model = sm.OLS(hNextFlat,regressors)
    health_results = health_model.fit()
    AvgHealthResidualByIncWealth = np.reshape(np.concatenate([[0.,0.,0.],health_results.params[-22:]]),(5,5))
    #print(health_results.summary())
    
    # Make flattened arrays of age, health, sex, and income and wealth quintiles (for OOP regression)
    Temp = np.logical_and(Useable,np.logical_not(np.isnan(m_data)))
    OOPflat = m_data[Temp]
    HealthFlat = h_data[Temp]
    HealthSqFlat = HealthFlat**2
    AgeFlat = age_data[Temp]
    AgeSqFlat = AgeFlat**2
    SexFlat = np.tile(np.reshape(sex_data,(1,obs)),(8,1))[Temp]
    IQflat = np.tile(np.reshape(inc_quint_data,(1,obs)),(8,1))[Temp]
    WQflat = np.tile(np.reshape(wealth_quint_data,(1,obs)),(8,1))[Temp]

    # Make flat arrays with inc-wealth indicators (for OOP regression)
    IWQboolFlat = np.zeros((25,OOPflat.size))
    k = 0
    for i in range(5):
        right_inc = IQflat == i+1
        for j in range(5):
            right_wealth = WQflat == j+1
            these = np.logical_and(right_inc,right_wealth)
            IWQboolFlat[k,these] = 1.0
            k += 1

    # Make flat arrays with income and wealth indicators, separately
    IQboolFlat = np.zeros((5,OOPflat.size))
    for i in range(5):
        IQboolFlat[i,:] = IQflat == i+1
    WQboolFlat = np.zeros((5,OOPflat.size))
    for j in range(5):
        WQboolFlat[j,:] = WQflat == j+1
    
    # Run a basic regression to predict OOP medical spending
    #regressors = np.transpose(np.vstack([np.ones_like(HealthFlat),SexFlat,HealthFlat,HealthSqFlat,AgeFlat,AgeSqFlat,IQboolFlat[1:,:],WQboolFlat[1:,:]]))
    regressors = np.transpose(np.vstack([np.ones_like(HealthFlat),SexFlat,HealthFlat,HealthSqFlat,AgeFlat,AgeSqFlat,IWQboolFlat[3:,:]]))
    OOP_model = sm.OLS(OOPflat,regressors)
    OOP_results = OOP_model.fit()
    #AvgOOPResidualByIncWealth = np.reshape(np.concatenate([[0.],OOP_results.params[6:10],[0.],OOP_results.params[10:]]),(2,5))
    AvgOOPResidualByIncWealth = np.reshape(np.concatenate([[0.,0.,0.],OOP_results.params[-22:]]),(5,5))
    #print(OOP_results.summary())
    
    
    # Make boolean array of income quintiles for the data
    IncQuint = np.tile(np.reshape(inc_quint_data,(1,obs)),(8,1))
    IncQuintBoolArray = np.zeros((8,obs,5),dtype=bool)
    for j in range(5):
        right_quint = IncQuint == j+1
        IncQuintBoolArray[:,:,j] = right_quint
        
    # Make boolean array of wealth quintiles for the data
    WealthQuint = np.tile(np.reshape(wealth_quint_data,(1,obs)),(8,1))
    WealthQuintBoolArray = np.zeros((8,obs,5),dtype=bool)
    for j in range(5):
        right_quint = WealthQuint == j+1
        WealthQuintBoolArray[:,:,j] = right_quint
        
    # Make boolean array of health tertiles for the data
    HealthTert = np.tile(np.reshape(health_tert_data,(1,obs)),(8,1))
    HealthTertBoolArray = np.zeros((8,obs,3),dtype=bool)
    for j in range(3):
        right_tert = HealthTert == j+1
        HealthTertBoolArray[:,:,j] = right_tert
        
    # Make boolean array of health quintiles for the data
    HealthQuint = np.tile(np.reshape(health_quint_data,(1,obs)),(8,1))
    HealthQuintBoolArray = np.zeros((8,obs,5),dtype=bool)
    for j in range(5):
        right_quint = HealthQuint == j+1
        HealthQuintBoolArray[:,:,j] = right_quint
        
    # Make a boolean array of sex for the data
    Sex = np.tile(np.reshape(sex_data,(1,obs)),(8,1))
    SexBoolArray = np.zeros((8,obs,2),dtype=bool)
    for j in range(2):
        right_sex = Sex == j
        SexBoolArray[:,:,j] = right_sex
    
    # Calculate median wealth by income quintile by age: 75
    # Calculate mean health status by income quintile by age: 75
    # Calculate mean IHS OOP medical spending by income quintile by age: 75
    WealthByIncAge = np.zeros((5,15))
    HealthByIncAge = np.zeros((5,15))
    OOPbyIncAge = np.zeros((5,15))
    IncAgeCellSize = np.zeros((5,15))
    for i in range(5):
        for a in range(15):
            these = np.logical_and(Useable,np.logical_and(AgeBoolArray[:,:,a],IncQuintBoolArray[:,:,i]))
            WealthByIncAge[i,a] = np.median(w_data[these])
            HealthByIncAge[i,a] = np.mean(h_data[these])
            OOPbyIncAge[i,a] = np.nanmean(arsinh(m_data[these]))
            IncAgeCellSize[i,a] = np.sum(these)
    OOPdiffByIncAge = OOPbyIncAge[1:,:] - np.tile(OOPbyIncAge[0,:],(4,1))
    OOPdiffByInc = np.mean(OOPdiffByIncAge,axis=1)*0.5
            
    WealthNorm = np.sum(WealthByIncAge*IncAgeCellSize)/np.sum(IncAgeCellSize)
    HealthNorm = np.sum(HealthByIncAge*IncAgeCellSize)/np.sum(IncAgeCellSize)
    
    # Calculate median wealth by income quintile by wealth quintile by age: 375
    # Calculate mean health status by income quintile by wealth quintile by age: 375
    # Calculate mean IHS OOP medical spending by income quintile by wealth quintile by age: 375
    WealthByIncWealthAge = np.zeros((5,5,15))
    HealthByIncWealthAge = np.zeros((5,5,15))
    OOPbyIncWealthAge = np.zeros((5,5,15))
    IncWealthAgeCellSize = np.zeros((5,5,15))
    for i in range(5):
        for j in range(5):
            for a in range(15):
                these = np.logical_and(np.logical_and(Useable,np.logical_and(AgeBoolArray[:,:,a],IncQuintBoolArray[:,:,i])),WealthQuintBoolArray[:,:,j])
                WealthByIncWealthAge[i,j,a] = np.median(w_data[these])
                HealthByIncWealthAge[i,j,a] = np.mean(h_data[these])
                OOPbyIncWealthAge[i,j,a] = np.nanmean(arsinh(m_data[these]))
                IncWealthAgeCellSize[i,j,a] = np.sum(these)
    
    # Calculate mean health status by sex by health tertile by age: 90
    # Calculate mean IHS OOP medical spending by sex by health tertile by age: 90
    # Calculate mortality probability by sex by health tertile by age: 90
    HealthBySexHealthAge = np.zeros((2,3,15))
    OOPbySexHealthAge = np.zeros((2,3,15))
    MortBySexHealthAge = np.zeros((2,3,15))
    SexHealthAgeCellSize = np.zeros((2,3,15))
    SexHealthAgeCellSizeMort = np.zeros((2,3,15))
    for s in range(2):
        for h in range(3):
            for a in range(15):
                these = np.logical_and(Useable,np.logical_and(SexBoolArray[:,:,s],np.logical_and(HealthTertBoolArray[:,:,h],AgeBoolArray[:,:,a])))
                HealthBySexHealthAge[s,h,a] = np.mean(h_data[these])
                OOPbySexHealthAge[s,h,a] = np.nanmean(arsinh(m_data[these]))
                SexHealthAgeCellSize[s,h,a] = np.sum(these)
                those = np.logical_and(MortUseable,np.logical_and(SexBoolArray[:,:,s],np.logical_and(HealthTertBoolArray[:,:,h],AgeBoolArray[:,:,a])))
                DeathCount = float(np.sum(np.logical_and(those,JustDied)))
                MortBySexHealthAge[s,h,a] = DeathCount/float(np.sum(those))
                SexHealthAgeCellSizeMort[s,h,a] = np.sum(those)
                
    # Calculate mean OOP by age-sex
    for s in range(2):
        for a in range(15):
            these = np.logical_and(Useable,np.logical_and(SexBoolArray[:,:,s],AgeBoolArray[:,:,a]))
            OOPbySexHealthAge[1,s,a] = np.nanmean(arsinh(m_data[these]))
    OOPbySexHealthAge[1,2,:] = 0.0 # Delete old moments
    
    # Calculate mean OOP medical spending by age: 15
    # Calculate stdev OOP medical spending by age: 15
    # Calculate mortality probability by age: 15
    # Calculate stdev delta health by age: 15
    OOPbyAge = np.zeros(15)
    StDevOOPbyAge = np.zeros(15)
    MortByAge = np.zeros(15)
    StDevDeltaHealthByAge = np.zeros(15)
    AgeCellSize = np.zeros(15)
    AgeCellSizeMort = np.zeros(15)
    AgeCellSizeHealthDelta = np.zeros(15)
    for a in range(15):
        these = np.logical_and(Useable,AgeBoolArray[:,:,a])
        OOPbyAge[a] = np.nanmean(arsinh(m_data[these]))
        StDevOOPbyAge[a] = np.nanstd(arsinh(m_data[these]))
        thise = np.logical_and(HealthDeltaUseable,AgeBoolArray[:,:,a])
        StDevDeltaHealthByAge[a] = np.nanstd(HealthDelta[thise])
        AgeCellSize[a] = np.sum(these)
        AgeCellSizeHealthDelta[a] = np.sum(thise)
        those = np.logical_and(MortUseable,AgeBoolArray[:,:,a])
        DeathCount = float(np.sum(np.logical_and(those,JustDied)))
        MortByAge[a] = DeathCount/float(np.sum(those))
        AgeCellSizeMort[a] = np.sum(those)
    
    OOPnorm = np.dot(OOPbyAge,AgeCellSize)/np.sum(AgeCellSize)
    StDevOOPnorm = np.dot(StDevOOPbyAge,AgeCellSize)/np.sum(AgeCellSize)
    MortNorm = np.dot(MortByAge,AgeCellSize)/np.sum(AgeCellSize)
    DeltaHealthNorm = np.dot(StDevDeltaHealthByAge,AgeCellSize)/np.sum(AgeCellSize)
    
    # Calculate stdev IHS OOP medical spending by health tertile by age: 45
    # Calculate stdev delta health by health tertile by age: 45
    StDevOOPbyHealthAge = np.zeros((3,15))
    StDevDeltaHealthByHealthAge = np.zeros((3,15))
    HealthAgeCellSize = np.zeros((3,15))
    HealthAgeCellSizeHealthDelta = np.zeros((3,15))
    for h in range(3):
        for a in range(15):
            these = np.logical_and(Useable,np.logical_and(HealthTertBoolArray[:,:,h],AgeBoolArray[:,:,a]))
            OOPbySexHealthAge[0,h,a] = np.nanmean(arsinh(m_data[these]))
            StDevOOPbyHealthAge[h,a] = np.nanstd(arsinh(m_data[these]))
            HealthAgeCellSize[h,a] = np.sum(these)
            thise = np.logical_and(HealthDeltaUseable,np.logical_and(HealthTertBoolArray[:,:,h],AgeBoolArray[:,:,a]))
            StDevDeltaHealthByHealthAge[h,a] = np.nanstd(HealthDelta[thise])
            HealthAgeCellSizeHealthDelta[h,a] = np.sum(thise)
            
    # Calculate mortality by health quintile by age: 75
    # Calculate health by health quintile by age: 75
    MortByHealthAge = np.zeros((5,15))
    HealthByHealthAge = np.zeros((5,15))
    for h in range(5):
        for a in range(15):
            those = np.logical_and(MortUseable,np.logical_and(HealthQuintBoolArray[:,:,h],AgeBoolArray[:,:,a]))
            DeathCount = float(np.sum(np.logical_and(those,JustDied)))
            MortByHealthAge[h,a] = DeathCount/float(np.sum(those))
            these = np.logical_and(Useable,np.logical_and(HealthQuintBoolArray[:,:,h],AgeBoolArray[:,:,a]))
            HealthByHealthAge[h,a] = np.mean(h_data[these])
            
    # Aggregate moments into a single vector
    all_moments = np.concatenate([
            OOPbyAge,
            StDevOOPbyAge,
            MortByAge,
            StDevDeltaHealthByAge,
            StDevOOPbyHealthAge.flatten(),
            StDevDeltaHealthByHealthAge.flatten(),
            HealthBySexHealthAge.flatten(),
            OOPbySexHealthAge.flatten(),
            MortBySexHealthAge.flatten(),
            WealthByIncAge.flatten(),
            HealthByIncAge.flatten(),
            OOPbyIncAge.flatten(),
            WealthByIncWealthAge.flatten(),
            HealthByIncWealthAge.flatten(),
            OOPbyIncWealthAge.flatten(),
            AvgHealthResidualByIncWealth.flatten(),
            AvgOOPResidualByIncWealth.flatten(),
            MortByHealthAge.flatten(),
            HealthByHealthAge.flatten()
            ])
    
    HealthProdPreEstMoments = np.concatenate([AvgHealthResidualByIncWealth.flatten(),OOPdiffByInc])
    
    # If this is not the last loop, store the moments for this loop in the array
    if b < data_bootstrap_count:
        BootstrappedResiduals[b,:] = np.concatenate([AvgHealthResidualByIncWealth.flatten()[3:],OOPdiffByInc])
        if np.any(np.isnan(all_moments)):
            BootstrapValidBool[b] = False
            valid_word = 'invalid'
        else:
            BootstrappedMoments[b,:] = all_moments
            BootstrapValidBool[b] = True
            valid_word = 'valid'
        print('Finished data bootstrap run ' + str(b+1) + ' of ' + str(data_bootstrap_count) + ', ' + valid_word)
    
# Make moment masking array
moment_mask = np.concatenate([
        np.ones(15)*moment_dummies[0],
        np.ones(15)*moment_dummies[1],
        np.ones(15)*moment_dummies[2],
        np.ones(15)*moment_dummies[3],
        np.ones(45)*moment_dummies[4],
        np.ones(45)*moment_dummies[5],
        np.ones(90)*moment_dummies[6],
        np.ones(75)*moment_dummies[7], # OOPbySexHealthAge[0,:,:] now has data by age-health, [1,0:2,:] has data for age-sex
        np.zeros(15), # Don't use these moments ever, they are blank
        np.ones(90)*moment_dummies[8],
        np.ones(75)*moment_dummies[9],
        np.ones(75)*moment_dummies[10],
        np.ones(75)*moment_dummies[11],
        np.ones(375)*moment_dummies[12],
        np.ones(375)*moment_dummies[13],
        np.ones(375)*moment_dummies[14],
        np.ones(25)*moment_dummies[15],
        np.ones(24)*moment_dummies[16],
        np.zeros(1)*moment_dummies[16],
        np.ones(75)*moment_dummies[17],
        np.ones(75)*moment_dummies[18]
        ])

# If the data moments were bootstrapped, calculate the optimal weighting matrix and save it to a file.
# Otherwise, try to read the weighting matrix from that file; if it doesn't exist, use the identity matrix.
if data_bootstrap_count > 0:
    # Calculate the covariance matrix of data moments
    BootstrappedMoments_valid = BootstrappedMoments[BootstrapValidBool,:]
    N = np.sum(BootstrapValidBool)
    VarVec = np.var(BootstrappedMoments_valid,axis=0)
    CovMatrix = np.cov(BootstrappedMoments_valid.transpose())
    print(str(N) + ' of ' + str(data_bootstrap_count) + ' bootstrap runs were valid.')
    moment_valid = VarVec > 1e-6 # Discard moments with zero or nearly zero variance
    moment_valid[645:675] = False # Turn off wealth moments for bottom two wealth quintiles of bottom income quintile
    moment_valid[720:735] = False # Turn off wealth moments for bottom wealth quintile of second income quintile
    moment_valid[1405:1410] = False # Turn off OOP moments for bottom wealth quintile of bottom income quintile after age 85
    moment_valid[1770:1773] = False # Turn off health residual moments for bottom three wealth quintiles of bottom income quintile
    moment_valid[1795:1798] = False # Turn off OOP residual moments for bottom three wealth quintiles of bottom income quintile
    valid_moment_N = np.sum(moment_valid)
    print(str(valid_moment_N) + ' of ' + str(moment_valid.size) + ' data moments were useable.')
    
    # For any moments with zero variation, temporarily remove them before inverting the covariance matrix,
    # then re-insert rows and columns of zeros after inverting.  These moments won't be used by the estimator,
    # even though *we are very confident in the data moments*.  The excluded moments are median wealth of
    # the poorest wealth-income groups, who always have zero assets.  The estimated model hits these moments
    # anyway, as the almost-poorest groups will have very little wealth, and the poorest have even less (zero).
    
    # Define moment groups for creating diagonal blocks of the weighting matrix
    moment_cuts = [0,15,30,45, # age moments
                   60,105, # health age moments
                   150,195,240,285,330,375, # health sex age moments
                   420,495,570, # health income age moments
                   645,720,795,870,945, # income wealth age moments: wealth
                   1020,1095,1170,1245,1320, # income wealth age moments: health
                   1395,1470,1545,1620,1695, # income wealth age moments: OOP
                   1770, # health residual moments
                   1795, # OOP residual moments
                   1820,1895, # health age moments: mort and health
                   moment_count] # end
                   
    # This vector indicates whether each block of moments should use the inverse of
    # the variance-covariance submatrix (True) or simply the inverse of the variance
    # elements as a diagonal matrix (False).  It is here to experiment with whether
    # moment covariances for mean OOP spending are driving estimation results.
    use_cov = np.ones(34,dtype=bool) 
    #use_cov[0] = False  # OOP by age
    #use_cov[8] = False  # OOP by health-age, women
    #use_cov[9] = False  # OOP by health-age, men
    #use_cov[14] = False # OOP by income-age
    weight_block_list = []
    
    # Loop through each block of moments and find the weighting matrix for that block
    for j in range(len(moment_cuts)-1):
        bot = moment_cuts[j]
        top = moment_cuts[j+1]
        which = np.zeros(moment_count,dtype=bool)
        which[bot:top] = True
        which = np.logical_and(which,moment_valid)
        CovMatrix_temp = CovMatrix[which,:][:,which]
        if use_cov[j]:
            weight_block_list.append(np.linalg.inv(CovMatrix_temp))
        else:
            weight_block_list.append(np.diag(np.diag(CovMatrix_temp)**(-1)))
    
    # Combine the blocks of weights into a diagonal matrix and re-insert rows and columns of zeros for omitted moments
    weighting_matrix_small = sp.linalg.block_diag(*weight_block_list)
    weighting_matrix_mid = np.zeros((moment_count,valid_moment_N))
    weighting_matrix_mid[moment_valid,:] = weighting_matrix_small
    weighting_matrix = np.zeros((moment_count,moment_count))
    weighting_matrix[:,moment_valid] = weighting_matrix_mid
    
    # Make a weighting matrix for the health production pre-estimation moments
    CovMatrixR = np.cov(BootstrappedResiduals.transpose())
    W = np.zeros((29,29))
    W[3:,3:] = np.linalg.inv(CovMatrixR)
    
    # Record the weighting matrix in a CSV file so we don't have to bootstrap the data every time
    with open('./Data/MomentWeights.txt','wb') as f:
        my_writer = csv.writer(f, delimiter = '\t')
        for i in range(weighting_matrix.shape[0]):
            my_writer.writerow(weighting_matrix[i,:])
        f.close()
     
    with open('./Data/PreEstWeights.txt','wb') as f:
        my_writer = csv.writer(f, delimiter = '\t')
        for i in range(W.shape[0]):
            my_writer.writerow(W[i,:])
        f.close()
    
        
else: # Try to read the weighting matrix from a CSV file if it wasn't just created
    try:
        infile = open('./Data/MomentWeights.txt','r')
        my_reader = csv.reader(infile,delimiter='\t')
        moment_weight_data = list(my_reader)
        infile.close()
        weighting_matrix = np.zeros((moment_count,moment_count))
        for i in range(moment_count):
            for j in range(moment_count):
                weighting_matrix[i,j] = float(moment_weight_data[i][j])
                
        infile = open('./Data/PreEstWeights.txt','r')
        my_reader = csv.reader(infile,delimiter='\t')
        preest_weight_data = list(my_reader)
        infile.close()
        W = np.zeros((29,29))
        for i in range(29):
            for j in range(29):
                W[i,j] = float(preest_weight_data[i][j])
    except:
        weighting_matrix = np.eye(moment_count)
        W = np.eye(25)
  
        
# Load in the absolute timepath of the relative price of care: 1977 to 2011
Years = np.arange(1977,2058)
AllYearPricePath = np.zeros(81) + np.nan
AllYearPricePath[0] = 54.6/58.7
AllYearPricePath[1]= 59.3/62.7
AllYearPricePath[2] = 64.8/68.5
AllYearPricePath[3] = 71.4/78.0
AllYearPricePath[4] = 78.600/87.2
AllYearPricePath[5] = 88.200/94.4
AllYearPricePath[6] = 97.900/97.9
AllYearPricePath[7] = 104.000/102.1
AllYearPricePath[8] = 110.200/105.7
AllYearPricePath[9] = 118.000/109.9
AllYearPricePath[10] = 126.700/111.4
AllYearPricePath[11] = 134.400/116.0
AllYearPricePath[12] = 143.800/121.2
AllYearPricePath[13] = 156.000/127.5
AllYearPricePath[14] = 171.000/134.7
AllYearPricePath[15] = 184.300/138.3
AllYearPricePath[16] = 196.400/142.8
AllYearPricePath[17] = 206.400/146.3
AllYearPricePath[18] = 216.600/150.5
AllYearPricePath[19] = 225.200/154.7
AllYearPricePath[20] = 231.800/159.4
AllYearPricePath[21] = 238.100/162.0
AllYearPricePath[22] = 246.500/164.7
AllYearPricePath[23] = 255.600/169.3
AllYearPricePath[24] = 267.200/175.6
AllYearPricePath[25] = 279.800/177.7
AllYearPricePath[26] = 292.700/182.6
AllYearPricePath[27] = 303.800/186.3
AllYearPricePath[28] = 316.900/191.6
AllYearPricePath[29] = 329.600/199.3
AllYearPricePath[30] = 343.596/203.379
AllYearPricePath[31] = 360.489/212.180
AllYearPricePath[32] = 369.832/211.903
AllYearPricePath[33] = 382.673/217.458
AllYearPricePath[34] = 393.843/221.062
MedInflation = np.log(AllYearPricePath[34]/AllYearPricePath[4])/30
AllYearPricePath[35:] = np.exp(MedInflation*(np.arange(35,81) - 34))*AllYearPricePath[34]

# Remap the all year price path to two year intervals
MedPriceHistory = np.zeros(40)
for t in range(40):
    MedPriceHistory[t] = (AllYearPricePath[2*t+1] + AllYearPricePath[2*t+2])/2
    
# Calculate mortality by income quintile by age (only used for model validation)
MortByIncAge = np.zeros((5,15))
for i in range(5):
    for a in range(15):
        those = np.logical_and(MortUseable,np.logical_and(IncQuintBoolArray[:,:,i],AgeBoolArray[:,:,a]))
        DeathCount = float(np.sum(np.logical_and(those,JustDied)))
        MortByIncAge[i,a] = DeathCount/float(np.sum(those))

        
