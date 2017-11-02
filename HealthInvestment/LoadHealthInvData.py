'''
This module imports data for the health investment estimation from text files
exported from Stata.
'''

import sys
import os
import csv
import numpy as np
sys.path.insert(0,'../')
sys.path.insert(0,'./Data/')

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
InitBoolArray = np.zeros((8,obs),dtype=bool)
InitBoolArray[t0,idx] = True

# Get initial data states
w_init = w_data[t0,idx]
h_init = h_data[t0,idx]

# Load the income profiles into memory
infile = open('./Data/IncProfilesNew.txt','r') 
my_reader = csv.reader(infile,delimiter='\t')
all_data = list(my_reader)
infile.close()
TypeCount = len(all_data) - 1

# Extract income and maximum wealth data into arrays
IncomeArray = np.zeros((TypeCount,26))
MaxWealth = np.zeros(TypeCount)
for i in range(TypeCount):
    j = i+1
    MaxWealth[i] = float(all_data[j][1])
    for t in range(21):
        IncomeArray[i,t] = float(all_data[j][t+2])
IncomeArray[:,21:] = np.tile(np.reshape(IncomeArray[:,20],(TypeCount,1)),(1,5))

# Make a "small" array of income profiles by sex and income quintile
IncomeArraySmall = np.zeros((10,26))
MaxWealthSmall = np.zeros(10)
for j in range(10):
    these = j + np.arange(0,150,10)
    IncomeArraySmall[j,:] = np.mean(IncomeArray[these,:],axis=0)
    MaxWealthSmall[j] = np.max(MaxWealth[these])

# Make a boolean array indicating when each agent is "born" in the data
turn65_t = cohort_data - 11
born_t = t0 - np.maximum(turn65_t,0)
BornBoolArray = np.zeros((25,obs),dtype=bool)
BornBoolArray[born_t,idx] = True

# Make an array of "ages" for each observation (plus boolean version)
age_data = np.tile(np.reshape(np.arange(8),(8,1)),(1,obs)) - np.tile(np.reshape(turn65_t,(1,obs)),(8,1))
AgeBoolArray = np.zeros((8,obs,15),dtype=bool)
for j in range(15):
    right_age = age_data == j+1
    AgeBoolArray[:,:,j] = right_age
    
# Make a boolean array of usable observations (for non-mortality moments)
BelowCohort16 = np.tile(np.reshape(cohort_data,(1,obs)),(8,1)) < 16
Alive = h_data > 0.
NotInit = np.logical_not(InitBoolArray)
Useable = np.logical_and(BelowCohort16,np.logical_and(Alive,NotInit))

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

# Calculate mean IHS OOP medical spending by age: 15
# Calculate stdev IHS OOP medical spending by age: 15
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

# Calculate stdev IHS OOP medical spending by health tertile by age: 45
# Calculate stdev delta health by health tertile by age: 45
StDevOOPbyHealthAge = np.zeros((3,15))
StDevDeltaHealthByHealthAge = np.zeros((3,15))
HealthAgeCellSize = np.zeros((3,15))
HealthAgeCellSizeHealthDelta = np.zeros((3,15))
for h in range(3):
    for a in range(15):
        these = np.logical_and(Useable,np.logical_and(HealthTertBoolArray[:,:,h],AgeBoolArray[:,:,a]))
        StDevOOPbyHealthAge[h,a] = np.nanstd(arsinh(m_data[these]))
        StDevDeltaHealthByHealthAge[h,a] = np.nan
        HealthAgeCellSize[h,a] = np.sum(these)
        thise = np.logical_and(HealthDeltaUseable,np.logical_and(HealthTertBoolArray[:,:,h],AgeBoolArray[:,:,a]))
        StDevDeltaHealthByHealthAge[h,a] = np.nanstd(HealthDelta[thise])
        HealthAgeCellSizeHealthDelta[h,a] = np.sum(thise)
        
        
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

        
