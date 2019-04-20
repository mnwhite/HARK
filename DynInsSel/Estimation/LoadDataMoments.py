'''
This file loads in data from the MEPS and SCF and calculates empirical moments.
It also resamples/booststraps the datasets to compute standard errors for the
empirical moments and produce a weighting matrix.  Alternatively, it can simply
load the previously computed weighting matrix from a file.
'''
import sys 
sys.path.insert(0,'../../')

import numpy as np
import matplotlib.pyplot as plt
import csv
import os
from HARKutilities import getPercentiles

# Set file locations and number of times to bootstrap data for empirical moment variances
bootstrap_count = 0
MEPS_data_filename = 'MEPSdataForDynInsSel.txt'
SCF_data_filename = 'SCFdataForDynInsSel.txt'
moment_weight_filename = 'MomentWeights.txt'

# Load the MEPS data into a CSV reader object
data_location = os.path.dirname(os.path.abspath(__file__))
f = open(data_location + '/' + MEPS_data_filename,'r')
MEPS_reader = csv.reader(f,delimiter='\t')
raw_MEPS_data = list(MEPS_reader)[1:]
f.close()
MEPS_obs = len(raw_MEPS_data)

# Load the SCF data into a CSV reader object
data_location = os.path.dirname(os.path.abspath(__file__))
f = open(data_location + '/' + SCF_data_filename,'r')
SCF_reader = csv.reader(f,delimiter='\t')
raw_SCF_data = list(SCF_reader)[1:]
f.close()
SCF_obs = len(raw_SCF_data)

# Initialize arrays to hold the MEPS data
Age_MEPSorig = np.zeros(MEPS_obs,dtype=int)
Health_MEPSorig = np.zeros(MEPS_obs,dtype=int)
Employed_MEPSorig = np.zeros(MEPS_obs,dtype=bool)
Offered_MEPSorig = np.zeros(MEPS_obs,dtype=bool)
Uninsured_MEPSorig = np.zeros(MEPS_obs,dtype=bool)
HasESI_MEPSorig = np.zeros(MEPS_obs,dtype=bool)
NoPremShare_MEPSorig = np.zeros(MEPS_obs,dtype=bool)
OOPmed_MEPSorig = np.zeros(MEPS_obs,dtype=float)
TotalMed_MEPSorig = np.zeros(MEPS_obs,dtype=float)
Income_MEPSorig = np.zeros(MEPS_obs,dtype=float)
Premiums_MEPSorig = np.zeros(MEPS_obs,dtype=float)
Weight_MEPSorig = np.zeros(MEPS_obs,dtype=float)

# Loop through each observation of the MEPS data and put it in the arrays
for i in range(MEPS_obs):
    this_ob = raw_MEPS_data[i]
    Age_MEPSorig[i] = int(float(this_ob[0]))
    Health_MEPSorig[i] = int(float(this_ob[1]))
    Employed_MEPSorig[i] = bool(float(this_ob[2]))
    Offered_MEPSorig[i] = bool(float(this_ob[3]))
    Uninsured_MEPSorig[i] = bool(float(this_ob[4]))
    HasESI_MEPSorig[i] = bool(float(this_ob[5]))
    NoPremShare_MEPSorig[i] = bool(float(this_ob[6]))
    OOPmed_MEPSorig[i] = float(this_ob[7])
    TotalMed_MEPSorig[i] = float(this_ob[8])
    Income_MEPSorig[i] = float(this_ob[9])
    Premiums_MEPSorig[i] = float(this_ob[10])
    Weight_MEPSorig[i] = float(this_ob[11])
    
# Initialize arrays to hold the MEPS data
Age_SCForig = np.zeros(SCF_obs,dtype=int)
Employed_SCForig = np.zeros(SCF_obs,dtype=bool)
HasESI_SCForig = np.zeros(SCF_obs,dtype=bool)
NetWorth_SCForig = np.zeros(SCF_obs,dtype=float)
Income_SCForig = np.zeros(SCF_obs,dtype=float)
Weight_SCForig = np.zeros(SCF_obs,dtype=float)

# Loop through each observation of the SCF data and put it in the arrays
for i in range(SCF_obs):
    this_ob = raw_SCF_data[i]
    Age_SCForig[i] = int(float(this_ob[0]))
    Employed_SCForig[i] = bool(float(this_ob[1]))
    HasESI_SCForig[i] = bool(float(this_ob[2]))
    NetWorth_SCForig[i] = float(this_ob[3])
    Income_SCForig[i] = float(this_ob[4])
    Weight_SCForig[i] = float(this_ob[5])
    
# Initialize arrays to hold empirical moments based on the MEPS and SCF
MeanLogOOPmedByAge = np.zeros(60)   # 0:60
MeanLogTotalMedByAge = np.zeros(60) # 60:120
StdevLogOOPmedByAge = np.zeros(60)  # 120:180
StdevLogTotalMedByAge = np.zeros(60)# 180:240
OOPshareByAge = np.zeros(60)        # 240:300
ESIinsuredRateByAge = np.zeros(40)  # 300:340
IMIinsuredRateByAge = np.zeros(40)  # 340:380
MeanESIpremiumByAge = np.zeros(40)  # 380:420
StdevESIpremiumByAge = np.zeros(40) # 420:460
NoPremShareRateByAge = np.zeros(40) # 460:500

MeanLogOOPmedByAgeHealth = np.zeros((12,5))   # 500:560
MeanLogTotalMedByAgeHealth = np.zeros((12,5)) # 560:620
StdevLogOOPmedByAgeHealth = np.zeros((12,5))  # 620:680
StdevLogTotalMedByAgeHealth = np.zeros((12,5))# 680:740
OOPshareByAgeHealth = np.zeros((12,5))        # 740:800
ESIinsuredRateByAgeHealth = np.zeros((8,5))   # 800:840
IMIinsuredRateByAgeHealth = np.zeros((8,5))   # 840:880
MeanESIpremiumByAgeHealth = np.zeros((8,5))   # 880:920
StdevESIpremiumByAgeHealth = np.zeros((8,5))  # 920:960
NoPremShareRateByAgeHealth = np.zeros((8,5))  # 960:1000

MeanLogOOPmedByAgeIncome = np.zeros((8,5))    # 1000:1040
MeanLogTotalMedByAgeIncome = np.zeros((8,5))  # 1040:1080
StdevLogOOPmedByAgeIncome = np.zeros((8,5))   # 1080:1120
StdevLogTotalMedByAgeIncome = np.zeros((8,5)) # 1120:1160
OOPshareByAgeIncome = np.zeros((8,5))         # 1160:1200
ESIinsuredRateByAgeIncome = np.zeros((8,5))   # 1200:1240
IMIinsuredRateByAgeIncome = np.zeros((8,5))   # 1240:1280
MeanESIpremiumByAgeIncome = np.zeros((8,5))   # 1280:1320
StdevESIpremiumByAgeIncome = np.zeros((8,5))  # 1320:1360
NoPremShareRateByAgeIncome = np.zeros((8,5))  # 1360:1400

MedianWealthRatioByAge = np.zeros(40)         # 1400:1440
MedianWealthRatioByAgeIncome = np.zeros((8,5))# 1440:1480

moment_count = 1480
age_group_limits = [[25,29],[30,34],[35,39],[40,44],[45,49],[50,54],[55,59],[60,64],[65,69],[70,74],[75,79],[80,84]]

# Initialize a giant array to hold bootstrapped empirical moments
if bootstrap_count > 0:
    BootstrappedMomentArray = np.zeros((moment_count,bootstrap_count))

# Bootstrap the data, resampling from the MEPS
b = 0
while b <= bootstrap_count:
    
    if b == bootstrap_count: # If this is the final pass, use the MEPS data as is
        idx = np.arange(MEPS_obs,dtype=int)
    else: # Otherwise, randomly resample from the MEPS
        idx = np.floor(np.random.rand(MEPS_obs)*MEPS_obs).astype(int)
    
    # Extract this resampling of the MEPS
    Age = Age_MEPSorig[idx]
    Health = Health_MEPSorig[idx]
    Employed = Employed_MEPSorig[idx]
    Offered = Offered_MEPSorig[idx]
    Uninsured = Uninsured_MEPSorig[idx]
    HasESI = HasESI_MEPSorig[idx]
    NoPremShare = NoPremShare_MEPSorig[idx]
    OOPmed = OOPmed_MEPSorig[idx]
    TotalMed = TotalMed_MEPSorig[idx]
    Income = Income_MEPSorig[idx]
    Premiums = Premiums_MEPSorig[idx]*12 # Convert monthly to annual
    Weight = Weight_MEPSorig[idx]
    
    # Calculate some simple objects from the data
    OOPnonzero = OOPmed > 0.
    TotalNonzero = TotalMed > 0.
    LogOOPmed = np.log(OOPmed)
    LogTotalMed = np.log(TotalMed)
    PremSeen = Premiums >= 0.
    
    # Make a boolean array of health states
    HealthBoolArray = np.zeros((MEPS_obs,5),dtype=bool)
    for h in range(5):
        these = Health == (h+1)
        HealthBoolArray[these,h] = True
        
    # Initialize a boolean array of income quintiles
    IncQuintBoolArray = np.zeros((MEPS_obs,5),dtype=bool)
    
    # Loop through each age and calculate moments
    for j in range(60):
        if j < 40:
            these = np.logical_and(Age == (j + 25), Employed)
        else:
            these = Age == (j + 25)
            
        # Fill in income quintile data for this age
        IncomeTemp = Income[these]
        WeightTemp = Weight[these]
        WeightTemp = WeightTemp/np.sum(WeightTemp)
        IncPctiles = getPercentiles(IncomeTemp,weights=WeightTemp,percentiles=[0.2,0.4,0.6,0.8],presorted=False)
        IncQuintBoolArray[these,0] = IncomeTemp < IncPctiles[0]
        IncQuintBoolArray[these,1] = np.logical_and(IncomeTemp >= IncPctiles[0], IncomeTemp < IncPctiles[1]) 
        IncQuintBoolArray[these,2] = np.logical_and(IncomeTemp >= IncPctiles[1], IncomeTemp < IncPctiles[2]) 
        IncQuintBoolArray[these,3] = np.logical_and(IncomeTemp >= IncPctiles[2], IncomeTemp < IncPctiles[3])
        IncQuintBoolArray[these,4] = IncomeTemp >= IncPctiles[3]
        
        # Get mean and stdev of log non-zero OOP medical expenses
        those = np.logical_and(these, OOPnonzero)
        WeightTemp = Weight[those]
        WeightTemp = WeightTemp/np.sum(WeightTemp)
        MeanLogOOPmed_j = np.dot(LogOOPmed[those],WeightTemp)
        MeanLogOOPmedByAge[j] = MeanLogOOPmed_j
        LogOOPmed_errsq = (LogOOPmed[those] - MeanLogOOPmed_j)**2
        VarLogOOPmed_j = np.dot(LogOOPmed_errsq,WeightTemp)
        StdevLogOOPmed_j = np.sqrt(VarLogOOPmed_j)
        StdevLogOOPmedByAge[j] = StdevLogOOPmed_j
        
        # Get mean and stdev of log non-zero total medical expenses
        those = np.logical_and(these, TotalNonzero)
        WeightTemp = Weight[those]
        WeightTemp = WeightTemp/np.sum(WeightTemp)
        MeanLogTotalMed_j = np.dot(LogTotalMed[those],WeightTemp)
        MeanLogTotalMedByAge[j] = MeanLogTotalMed_j
        LogTotalMed_errsq = (LogTotalMed[those] - MeanLogTotalMed_j)**2
        VarLogTotalMed_j = np.dot(LogTotalMed_errsq,WeightTemp)
        StdevLogTotalMed_j = np.sqrt(VarLogTotalMed_j)
        StdevLogTotalMedByAge[j] = StdevLogTotalMed_j
        
        # Get out-of-pocket medical spending share
        if j < 40:
            those = np.logical_and(these, HasESI)
        else:
            those = these
        WeightTemp = Weight[those]
        WeightTemp = WeightTemp/np.sum(WeightTemp)
        OOPmedSum = np.dot(OOPmed[those], WeightTemp)
        TotalMedSum = np.dot(TotalMed[those], WeightTemp)
        OOPshareByAge[j] = OOPmedSum/TotalMedSum
        
        # Get insured rate for the ESI and IMI populations, mean/stdev of ESI
        # out-of-pocket premiums, and rate of paying full price for ESI.
        if j < 40:
            those = np.logical_and(these, Offered)
            WeightTemp = Weight[those]
            WeightTemp = WeightTemp/np.sum(WeightTemp)
            ESIinsuredRateByAge[j] = np.dot(np.logical_not(Uninsured[those]),WeightTemp)
            
            those = np.logical_and(these, np.logical_not(Offered))
            WeightTemp = Weight[those]
            WeightTemp = WeightTemp/np.sum(WeightTemp)
            IMIinsuredRateByAge[j] = np.dot(np.logical_not(Uninsured[those]),WeightTemp)
            
            those = np.logical_and(these, PremSeen)
            WeightTemp = Weight[those]
            WeightTemp = WeightTemp/np.sum(WeightTemp)
            MeanESIpremium_j = np.dot(Premiums[those],WeightTemp)
            ESIpremium_errsq = (Premiums[those] - MeanESIpremium_j)**2
            VarESIpremium_j = np.dot(ESIpremium_errsq, WeightTemp)
            StdevESIpremium_j = np.sqrt(VarESIpremium_j)
            MeanESIpremiumByAge[j] = MeanESIpremium_j
            StdevESIpremiumByAge[j] = StdevESIpremium_j
            
            those = np.logical_and(np.logical_and(these, HasESI), PremSeen)
            WeightTemp = Weight[those]
            WeightTemp = WeightTemp/np.sum(WeightTemp)
            NoPremShareRateByAge[j] = np.dot(NoPremShare[those], WeightTemp)
            
    # Loop through each age group
    for g in range(12):
        age_min = age_group_limits[g][0]
        age_max = age_group_limits[g][1]
        THESE = np.logical_and(Age >= age_min, Age <= age_max)
        if g < 8:
            THESE = np.logical_and(THESE, Employed)
            
        # Loop through each health status
        for h in range(5):
            these = np.logical_and(THESE, HealthBoolArray[:,h])
            
            # Get mean and stdev of log non-zero OOP medical expenses
            those = np.logical_and(these, OOPnonzero)
            WeightTemp = Weight[those]
            WeightTemp = WeightTemp/np.sum(WeightTemp)
            MeanLogOOPmed_gh = np.dot(LogOOPmed[those],WeightTemp)
            MeanLogOOPmedByAgeHealth[g,h] = MeanLogOOPmed_gh
            LogOOPmed_errsq = (LogOOPmed[those] - MeanLogOOPmed_gh)**2
            VarLogOOPmed_gh = np.dot(LogOOPmed_errsq,WeightTemp)
            StdevLogOOPmed_gh = np.sqrt(VarLogOOPmed_gh)
            StdevLogOOPmedByAgeHealth[g,h] = StdevLogOOPmed_gh
            
            # Get mean and stdev of log non-zero total medical expenses
            those = np.logical_and(these, TotalNonzero)
            WeightTemp = Weight[those]
            WeightTemp = WeightTemp/np.sum(WeightTemp)
            MeanLogTotalMed_gh = np.dot(LogTotalMed[those],WeightTemp)
            MeanLogTotalMedByAgeHealth[g,h] = MeanLogTotalMed_gh
            LogTotalMed_errsq = (LogTotalMed[those] - MeanLogTotalMed_gh)**2
            VarLogTotalMed_gh = np.dot(LogTotalMed_errsq,WeightTemp)
            StdevLogTotalMed_gh = np.sqrt(VarLogTotalMed_gh)
            StdevLogTotalMedByAgeHealth[g,h] = StdevLogTotalMed_gh
            
            # Get out-of-pocket medical spending share
            if g < 8:
                those = np.logical_and(these, HasESI)
            else:
                those = these
            WeightTemp = Weight[those]
            WeightTemp = WeightTemp/np.sum(WeightTemp)
            OOPmedSum = np.dot(OOPmed[those], WeightTemp)
            TotalMedSum = np.dot(TotalMed[those], WeightTemp)
            OOPshareByAgeHealth[g,h] = OOPmedSum/TotalMedSum
            
            # Get insured rate for the ESI and IMI populations, mean/stdev of ESI
            # out-of-pocket premiums, and rate of paying full price for ESI.
            if g < 8:
                those = np.logical_and(these, Offered)
                WeightTemp = Weight[those]
                WeightTemp = WeightTemp/np.sum(WeightTemp)
                ESIinsuredRateByAgeHealth[g,h] = np.dot(np.logical_not(Uninsured[those]),WeightTemp)
                
                those = np.logical_and(these, np.logical_not(Offered))
                WeightTemp = Weight[those]
                WeightTemp = WeightTemp/np.sum(WeightTemp)
                IMIinsuredRateByAgeHealth[g,h] = np.dot(np.logical_not(Uninsured[those]),WeightTemp)
                
                those = np.logical_and(these, PremSeen)
                WeightTemp = Weight[those]
                WeightTemp = WeightTemp/np.sum(WeightTemp)
                MeanESIpremium_gh = np.dot(Premiums[those],WeightTemp)
                ESIpremium_errsq = (Premiums[those] - MeanESIpremium_gh)**2
                VarESIpremium_gh = np.dot(ESIpremium_errsq, WeightTemp)
                StdevESIpremium_gh = np.sqrt(VarESIpremium_gh)
                MeanESIpremiumByAgeHealth[g,h] = MeanESIpremium_gh
                StdevESIpremiumByAgeHealth[g,h] = StdevESIpremium_gh
                
                those = np.logical_and(np.logical_and(these, HasESI), PremSeen)
                WeightTemp = Weight[those]
                WeightTemp = WeightTemp/np.sum(WeightTemp)
                NoPremShareRateByAgeHealth[g,h] = np.dot(NoPremShare[those], WeightTemp)
                
        # Loop through each income quintile
        for i in range(5):
            if g >= 8:
                continue
            
            these = np.logical_and(THESE, IncQuintBoolArray[:,i])
            
            # Get mean and stdev of log non-zero OOP medical expenses
            those = np.logical_and(these, OOPnonzero)
            WeightTemp = Weight[those]
            WeightTemp = WeightTemp/np.sum(WeightTemp)
            MeanLogOOPmed_gi = np.dot(LogOOPmed[those],WeightTemp)
            MeanLogOOPmedByAgeIncome[g,i] = MeanLogOOPmed_gi
            LogOOPmed_errsq = (LogOOPmed[those] - MeanLogOOPmed_gi)**2
            VarLogOOPmed_gi = np.dot(LogOOPmed_errsq,WeightTemp)
            StdevLogOOPmed_gi = np.sqrt(VarLogOOPmed_gi)
            StdevLogOOPmedByAgeIncome[g,i] = StdevLogOOPmed_gi
            
            # Get mean and stdev of log non-zero total medical expenses
            those = np.logical_and(these, TotalNonzero)
            WeightTemp = Weight[those]
            WeightTemp = WeightTemp/np.sum(WeightTemp)
            MeanLogTotalMed_gi = np.dot(LogTotalMed[those],WeightTemp)
            MeanLogTotalMedByAgeIncome[g,i] = MeanLogTotalMed_gi
            LogTotalMed_errsq = (LogTotalMed[those] - MeanLogTotalMed_gi)**2
            VarLogTotalMed_gi = np.dot(LogTotalMed_errsq,WeightTemp)
            StdevLogTotalMed_gi = np.sqrt(VarLogTotalMed_gi)
            StdevLogTotalMedByAgeIncome[g,i] = StdevLogTotalMed_gi
            
            # Get out-of-pocket medical spending share
            those = np.logical_and(these, HasESI)
            WeightTemp = Weight[those]
            WeightTemp = WeightTemp/np.sum(WeightTemp)
            OOPmedSum = np.dot(OOPmed[those], WeightTemp)
            TotalMedSum = np.dot(TotalMed[those], WeightTemp)
            OOPshareByAgeIncome[g,i] = OOPmedSum/TotalMedSum
            
            # Get insured rate for the ESI and IMI populations, mean/stdev of ESI
            # out-of-pocket premiums, and rate of paying full price for ESI.
            those = np.logical_and(these, Offered)
            WeightTemp = Weight[those]
            WeightTemp = WeightTemp/np.sum(WeightTemp)
            ESIinsuredRateByAgeIncome[g,i] = np.dot(np.logical_not(Uninsured[those]),WeightTemp)
            
            those = np.logical_and(these, np.logical_not(Offered))
            WeightTemp = Weight[those]
            WeightTemp = WeightTemp/np.sum(WeightTemp)
            IMIinsuredRateByAgeIncome[g,i] = np.dot(np.logical_not(Uninsured[those]),WeightTemp)
            
            those = np.logical_and(these, PremSeen)
            WeightTemp = Weight[those]
            WeightTemp = WeightTemp/np.sum(WeightTemp)
            MeanESIpremium_gi = np.dot(Premiums[those],WeightTemp)
            ESIpremium_errsq = (Premiums[those] - MeanESIpremium_gi)**2
            VarESIpremium_gi = np.dot(ESIpremium_errsq, WeightTemp)
            StdevESIpremium_gi = np.sqrt(VarESIpremium_gi)
            MeanESIpremiumByAgeIncome[g,i] = MeanESIpremium_gi
            StdevESIpremiumByAgeIncome[g,i] = StdevESIpremium_gi
            
            those = np.logical_and(np.logical_and(these, HasESI), PremSeen)
            WeightTemp = Weight[those]
            WeightTemp = WeightTemp/np.sum(WeightTemp)
            NoPremShareRateByAgeIncome[g,i] = np.dot(NoPremShare[those], WeightTemp)
    
    # Now resample from the SCF    
    if b == bootstrap_count: # If this is the final pass, use the SCF data as is
        idx = np.arange(SCF_obs,dtype=int)
    else: # Otherwise, randomly resample from the SCF
        idx = np.floor(np.random.rand(SCF_obs)*SCF_obs).astype(int)
        
    # Extract this resampling of the SCF
    Age = Age_SCForig[idx]
    Employed = Employed_SCForig[idx]
    HasESI = HasESI_SCForig[idx]
    NetWorth = NetWorth_SCForig[idx]
    Income = Income_SCForig[idx]
    Weight = Weight_SCForig[idx]
    WealthRatio = NetWorth/Income
    
    # Initialize the income quintile boolean array for the SCF
    IncQuintBoolArray = np.zeros((SCF_obs,5),dtype=bool)
    
    # Loop through each age and calculate median wealth ratio
    for j in range(40):
        these = np.logical_and(Age == (j + 25), Employed)
        
        WeightTemp = Weight[these]
        WeightTemp = WeightTemp/np.sum(WeightTemp)
        MedianWealthRatioByAge[j] = getPercentiles(WealthRatio[these],weights=WeightTemp)
        
        # Fill in income quintile data for this age
        IncomeTemp = Income[these]
        WeightTemp = Weight[these]
        WeightTemp = WeightTemp/np.sum(WeightTemp)
        IncPctiles = getPercentiles(IncomeTemp,weights=WeightTemp,percentiles=[0.2,0.4,0.6,0.8],presorted=False)
        IncQuintBoolArray[these,0] = IncomeTemp < IncPctiles[0]
        IncQuintBoolArray[these,1] = np.logical_and(IncomeTemp >= IncPctiles[0], IncomeTemp < IncPctiles[1]) 
        IncQuintBoolArray[these,2] = np.logical_and(IncomeTemp >= IncPctiles[1], IncomeTemp < IncPctiles[2]) 
        IncQuintBoolArray[these,3] = np.logical_and(IncomeTemp >= IncPctiles[2], IncomeTemp < IncPctiles[3])
        IncQuintBoolArray[these,4] = IncomeTemp >= IncPctiles[3]
        
    # Loop through each age group
    for g in range(8):
        age_min = age_group_limits[g][0]
        age_max = age_group_limits[g][1]
        THESE = np.logical_and(np.logical_and(Age >= age_min, Age <= age_max), Employed)
        
        # Loop through each income quintile
        for i in range(5):
            these = np.logical_and(THESE,IncQuintBoolArray[:,i])
            
            WeightTemp = Weight[these]
            WeightTemp = WeightTemp/np.sum(WeightTemp)
            MedianWealthRatioByAgeIncome[g,i] = getPercentiles(WealthRatio[these],weights=WeightTemp)
            
    # Gather the SCF and MEPS moments into a list
    MomentList = [
            MeanLogOOPmedByAge,
            MeanLogTotalMedByAge,
            StdevLogOOPmedByAge,
            StdevLogTotalMedByAge,
            OOPshareByAge,
            ESIinsuredRateByAge,
            IMIinsuredRateByAge,
            MeanESIpremiumByAge,
            StdevESIpremiumByAge,
            NoPremShareRateByAge,
            MeanLogOOPmedByAgeHealth.flatten(),
            MeanLogTotalMedByAgeHealth.flatten(),
            StdevLogOOPmedByAgeHealth.flatten(),
            StdevLogTotalMedByAgeHealth.flatten(),
            OOPshareByAgeHealth.flatten(),
            ESIinsuredRateByAgeHealth.flatten(),
            IMIinsuredRateByAgeHealth.flatten(),
            MeanESIpremiumByAgeHealth.flatten(),
            StdevESIpremiumByAgeHealth.flatten(),
            NoPremShareRateByAgeHealth.flatten(),
            MeanLogOOPmedByAgeIncome.flatten(),
            MeanLogTotalMedByAgeIncome.flatten(),
            StdevLogOOPmedByAgeIncome.flatten(),
            StdevLogTotalMedByAgeIncome.flatten(),
            OOPshareByAgeIncome.flatten(),
            ESIinsuredRateByAgeIncome.flatten(),
            IMIinsuredRateByAgeIncome.flatten(),
            MeanESIpremiumByAgeIncome.flatten(),
            StdevESIpremiumByAgeIncome.flatten(),
            NoPremShareRateByAgeIncome.flatten(),
            MedianWealthRatioByAge,
            MedianWealthRatioByAgeIncome.flatten()
            ]
    
    # Combine the MEPS moments into a single vector
    data_moments = np.concatenate(MomentList)
    if (bootstrap_count > 0) and (b < bootstrap_count):
        BootstrappedMomentArray[:,b] = data_moments
        
    # Move to the next bootstrap
    if (bootstrap_count > 0) and (b < bootstrap_count) and (np.mod(b+1,10) == 0):
        print('Finished bootstrap #' + str(b+1) + ' of ' + str(bootstrap_count) + '.')
    elif (b == bootstrap_count):
        print('Calculated MEPS and SCF empirical moments.')
    b += 1
    
# If the data was bootstrapped, calculate the variance of each empirical moment
# and save it to the moment file.
if bootstrap_count > 0:
    EmpiricalMomentVariances = np.var(BootstrappedMomentArray,axis=1)
    moment_weights = EmpiricalMomentVariances**(-1)
    moment_weights[np.isinf(moment_weights)] = 0. # NoPremShareByAgeHealth has no variation for some poor health, young ages
    moment_weights[800] = 0. # This moment has *almost* no observations and no variation, and thus ends up with a weight of 10**32, yikes
    moment_weights = np.minimum(moment_weights,1e6) # Two other moments have *very little* variation and would get *insane* weights
    f = open(data_location + '/' + moment_weight_filename,'w')
    my_writer = csv.writer(f, delimiter = '\t')
    my_writer.writerow(moment_weights)
    f.close()
    print('Saved moment weighting file to disk.')
        
else: # If the data was not bootstrapped, try to read the moment weights from file
    try:
        f = open(data_location + '/' + moment_weight_filename,'r')
        my_reader = csv.reader(f, delimiter='\t')
        raw_weights = list(my_reader)[0]
        moment_weights = np.zeros(moment_count)
        for i in range(moment_count):
            moment_weights[i] = float(raw_weights[i])
        f.close()
        print('Loaded moment weights from file.')
    except:
        print('Unable to open moment weighting file!')
        
        
if __name__ == '__main__':
    os.chdir('..')
    os.chdir('Figures')
    
    OneYearAgeLong = np.arange(25,85)
    FiveYearAgeLong = np.arange(25,85,5) + 2.5
    OneYearAge = np.arange(25,65)
    FiveYearAge = np.arange(25,65,5) + 2.5
    
    plt.plot(OneYearAge,MeanLogOOPmedByAge[0:40],'.k')
    plt.plot(FiveYearAge,MeanLogOOPmedByAgeIncome)
    plt.xlabel('Age')
    plt.ylabel('Mean log OOP medical expenses')
    plt.legend(['Overall average','Bottom income quintile','Second income quintile','Third income quintile','Fourth income quintile','Top income quintile'],loc=0,fontsize=8)
    plt.savefig('MeanLogOOPmedByAgeIncome.pdf')
    plt.show()
    
    plt.plot(OneYearAge,MeanLogTotalMedByAge[0:40],'.k')
    plt.plot(FiveYearAge,MeanLogTotalMedByAgeIncome)
    plt.xlabel('Age')
    plt.ylabel('Mean log total medical expenses')
    plt.legend(['Overall average','Bottom income quintile','Second income quintile','Third income quintile','Fourth income quintile','Top income quintile'],loc=0,fontsize=8)
    plt.savefig('MeanLogTotalMedByAgeIncome.pdf')
    plt.show()
    
    plt.plot(OneYearAge,StdevLogOOPmedByAge[0:40],'.k')
    plt.plot(FiveYearAge,StdevLogOOPmedByAgeIncome)
    plt.xlabel('Age')
    plt.ylabel('Stdev log OOP medical expenses')
    plt.legend(['Overall average','Bottom income quintile','Second income quintile','Third income quintile','Fourth income quintile','Top income quintile'],loc=0,fontsize=8)
    plt.savefig('StdevLogOOPmedByAgeIncome.pdf')
    plt.show()
    
    plt.plot(OneYearAge,StdevLogTotalMedByAge[0:40],'.k')
    plt.plot(FiveYearAge,StdevLogTotalMedByAgeIncome)
    plt.xlabel('Age')
    plt.ylabel('Stdev log total medical expenses')
    plt.legend(['Overall average','Bottom income quintile','Second income quintile','Third income quintile','Fourth income quintile','Top income quintile'],loc=0,fontsize=8)
    plt.savefig('StdevLogTotalMedByAgeIncome.pdf')
    plt.show()
    
    plt.plot(OneYearAge,ESIinsuredRateByAge,'.k')
    plt.plot(FiveYearAge,ESIinsuredRateByAgeIncome)
    plt.xlabel('Age')
    plt.ylabel('ESI uptake rate')
    plt.legend(['Overall average','Bottom income quintile','Second income quintile','Third income quintile','Fourth income quintile','Top income quintile'],loc=0,fontsize=8)
    plt.savefig('InsuredRateESIByAgeIncome.pdf')
    plt.show()
    
    plt.plot(OneYearAge,IMIinsuredRateByAge,'.k')
    plt.plot(FiveYearAge,IMIinsuredRateByAgeIncome)
    plt.xlabel('Age')
    plt.ylabel('IMI insured rate')
    plt.legend(['Overall average','Bottom income quintile','Second income quintile','Third income quintile','Fourth income quintile','Top income quintile'],loc=0,fontsize=8)
    plt.savefig('InsuredRateIMIByAgeIncome.pdf')
    plt.show()
    
    plt.plot(OneYearAge,MeanESIpremiumByAge,'.k')
    plt.plot(FiveYearAge,MeanESIpremiumByAgeIncome)
    plt.xlabel('Age')
    plt.ylabel('Mean ESI out-of-pocket premium')
    plt.ylim([500,3500])
    plt.legend(['Overall average','Bottom income quintile','Second income quintile','Third income quintile','Fourth income quintile','Top income quintile'],loc=0,fontsize=8)
    plt.savefig('MeanPremiumByAgeIncome.pdf')
    plt.show()
    
    plt.plot(OneYearAge,StdevESIpremiumByAge,'.k')
    plt.plot(FiveYearAge,StdevESIpremiumByAgeIncome)
    plt.xlabel('Age')
    plt.ylabel('Stdev ESI out-of-pocket premium')
    plt.legend(['Overall average','Bottom income quintile','Second income quintile','Third income quintile','Fourth income quintile','Top income quintile'],loc=0,fontsize=8)
    plt.savefig('StdevPremiumByAgeIncome.pdf')
    plt.show()
    
    plt.plot(OneYearAge,NoPremShareRateByAge,'.k')
    plt.plot(FiveYearAge,NoPremShareRateByAgeIncome)
    plt.xlabel('Age')
    plt.ylabel('Pct ESI buyers with no employer contribution')
    plt.legend(['Overall average','Bottom income quintile','Second income quintile','Third income quintile','Fourth income quintile','Top income quintile'],loc=0,fontsize=8)
    plt.savefig('NoPremShareByAgeIncome.pdf')
    plt.show()
    
    plt.plot(OneYearAge,MedianWealthRatioByAge,'.k')
    plt.plot(FiveYearAge,MedianWealthRatioByAgeIncome)
    plt.xlabel('Age')
    plt.ylabel('Median wealth-to-income ratio')
    plt.legend(['Overall average','Bottom income quintile','Second income quintile','Third income quintile','Fourth income quintile','Top income quintile'],loc=0,fontsize=8)
    plt.savefig('WealthRatioByAgeIncome.pdf')
    plt.show()
    
    plt.plot(OneYearAgeLong,MeanLogOOPmedByAge,'.k')
    plt.plot(FiveYearAgeLong,MeanLogOOPmedByAgeHealth)
    plt.xlabel('Age')
    plt.ylabel('Mean log OOP medical expenses')
    plt.legend(['Overall average','Poor health','Fair health','Good health','Very good health','Excellent health'],loc=0,fontsize=8)
    plt.savefig('MeanLogOOPmedByAgeHealth.pdf')
    plt.show()
    
    plt.plot(OneYearAgeLong,MeanLogTotalMedByAge,'.k')
    plt.plot(FiveYearAgeLong,MeanLogTotalMedByAgeHealth)
    plt.xlabel('Age')
    plt.ylabel('Mean log total medical expenses')
    plt.legend(['Overall average','Poor health','Fair health','Good health','Very good health','Excellent health'],loc=0,fontsize=8)
    plt.savefig('MeanLogTotalMedByAgeHealth.pdf')
    plt.show()
    
    plt.plot(OneYearAgeLong,OOPshareByAge,'.k')
    plt.xlabel('Age')
    plt.ylabel('Out-of-pocket medical spending share')
    plt.ylim([0.0,0.25])
    plt.savefig('OOPshareByAge.pdf')
    plt.show()
    
    plt.plot(OneYearAgeLong,StdevLogOOPmedByAge,'.k')
    plt.plot(FiveYearAgeLong,StdevLogOOPmedByAgeHealth)
    plt.xlabel('Age')
    plt.ylabel('Stdev log OOP medical expenses')
    plt.legend(['Overall average','Poor health','Fair health','Good health','Very good health','Excellent health'],loc=0,fontsize=8)
    plt.savefig('StdevLogOOPmedByAgeHealth.pdf')
    plt.show()
    
    plt.plot(OneYearAgeLong,StdevLogTotalMedByAge,'.k')
    plt.plot(FiveYearAgeLong,StdevLogTotalMedByAgeHealth)
    plt.xlabel('Age')
    plt.ylabel('Stdev log total medical expenses')
    plt.legend(['Overall average','Poor health','Fair health','Good health','Very good health','Excellent health'],loc=0,fontsize=8)
    plt.savefig('StdevLogTotalMedByAgeHealth.pdf')
    plt.show()
    
    plt.plot(OneYearAge,ESIinsuredRateByAge,'.k')
    plt.plot(FiveYearAge,ESIinsuredRateByAgeHealth)
    plt.xlabel('Age')
    plt.ylabel('ESI uptake rate')
    plt.legend(['Overall average','Poor health','Fair health','Good health','Very good health','Excellent health'],loc=0,fontsize=8)
    plt.savefig('InsuredRateByAgeHealth.pdf')
    plt.show()
    
    plt.plot(OneYearAge,MeanESIpremiumByAge,'.k')
    plt.plot(FiveYearAge,MeanESIpremiumByAgeHealth)
    plt.xlabel('Age')
    plt.ylabel('Mean OOP premium')
    plt.ylim([500,3500])
    plt.legend(['Overall average','Poor health','Fair health','Good health','Very good health','Excellent health'],loc=0,fontsize=8)
    plt.savefig('MeanPremiumByAgeHealth.pdf')
    plt.show()
    
    plt.plot(OneYearAge,StdevESIpremiumByAge,'.k')
    plt.plot(FiveYearAge,StdevESIpremiumByAgeHealth)
    plt.xlabel('Age')
    plt.ylabel('Stdev OOP premium')
    plt.legend(['Overall average','Poor health','Fair health','Good health','Very good health','Excellent health'],loc=0,fontsize=8)
    plt.savefig('StdevPremiumByAgeHealth.pdf')
    plt.show()
    
    plt.plot(OneYearAge,NoPremShareRateByAge,'.k')
    plt.plot(FiveYearAge,NoPremShareRateByAgeHealth)
    plt.xlabel('Age')
    plt.ylabel('Pct ESI buyers with no employer contribution')
    plt.legend(['Overall average','Poor health','Fair health','Good health','Very good health','Excellent health'],loc=0,fontsize=8)
    plt.savefig('NoPremShareByAgeHealth.pdf')
    plt.show()
    
    plt.plot(OneYearAge,NoPremShareRateByAge,'.k')
    plt.xlabel('Age')
    plt.ylabel('Pct ESI buyers with no employer contribution')
    plt.savefig('NoPremShareByAge.pdf')
    plt.show()


