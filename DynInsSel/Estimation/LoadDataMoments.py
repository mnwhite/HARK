'''
This module unpacks data moments as calculated in the MEPS, HRS, and SCF, loading them as arrays.
'''
import numpy as np
import csv
import os

# Choose which classes of moments will actually be used in estimation
use_data_weights = False # Whether to use moment weights from data or just all ones
MomentBools = np.array([
               True,  #WealthRatioByAge
               True,  #MeanLogTotalMedByAge
               False, #StdevLogTotalMedByAge
               False, #InsuredRateByAge
               False, #NoPremShareRateByAge
               False, #MeanPremiumByAge
               False, #StdevPremiumByAge
               False, #MeanLogTotalMedByAgeHealth
               False, #StdevLogTotalMedByAgeHealth
               False, #WealthRatioByAgeIncome
               False, #MeanLogTotalMedByAgeIncome
               False, #StdevLogTotalMedByAgeIncome
               False, #InsuredRateByAgeIncome
               False, #MeanPremiumByAgeIncome
               False, #MeanLogOOPmedByAge
               False, #StdevLogOOPmedByAge
              ])

# Load the moments by one-year age groups into a CSV reader object
data_location = os.path.dirname(os.path.abspath(__file__))
f = open(data_location + '\MomentsByAge.txt','r')
moment_reader = csv.reader(f,delimiter='\t')
raw_moments = list(moment_reader)
f.close()

# Store the moments by one-year age groups in arrays
OneYearAge = np.arange(25,65)
OneYearAgeLong = np.arange(25,85)
MeanLogOOPmedByAge = np.zeros(60) + np.nan
MeanLogTotalMedByAge = np.zeros(60) + np.nan
StdevLogOOPmedByAge = np.zeros(60) + np.nan
StdevLogTotalMedByAge = np.zeros(60) + np.nan
InsuredRateByAge = np.zeros(40) + np.nan
MeanPremiumByAge = np.zeros(40) + np.nan
StdevPremiumByAge = np.zeros(40) + np.nan
NoPremShareRateByAge = np.zeros(40) + np.nan
OOPshareByAge = np.zeros(60) + np.nan
OOPmomentWeightsByAge = np.zeros(60) + np.nan
totMomentWeightsByAge = np.zeros(60) + np.nan
premMomentWeightsByAge = np.zeros(40) + np.nan
otherMomentWeightsByAge = np.zeros(40) + np.nan
for j in range(60):
    MeanLogOOPmedByAge[j] = float(raw_moments[j][1])
    MeanLogTotalMedByAge[j] = float(raw_moments[j][2])
    StdevLogOOPmedByAge[j] = float(raw_moments[j][3])
    StdevLogTotalMedByAge[j] = float(raw_moments[j][4])
    OOPshareByAge[j] = float(raw_moments[j][9])
    OOPmomentWeightsByAge[j] = float(raw_moments[j][10])
    totMomentWeightsByAge[j] = float(raw_moments[j][11])
    if j < 40:
        InsuredRateByAge[j] = float(raw_moments[j][5])
        MeanPremiumByAge[j] = float(raw_moments[j][6])
        StdevPremiumByAge[j] = float(raw_moments[j][7])
        NoPremShareRateByAge[j] = float(raw_moments[j][8])
        premMomentWeightsByAge[j] = float(raw_moments[j][12])
        otherMomentWeightsByAge[j] = float(raw_moments[j][13])    

# Load the moments by five-year age groups and income quintile into a CSV reader object
data_location = os.path.dirname(os.path.abspath(__file__))
f = open(data_location + '\MomentsByAgeIncome.txt','r')
moment_reader = csv.reader(f,delimiter='\t')
raw_moments = list(moment_reader)
f.close()

# Store the moments by five-year age groups and income quintile in arrays
FiveYearAge = 5*np.arange(8) + 27
FiveYearAgeLong = 5*np.arange(12) + 27
MeanLogOOPmedByAgeIncome = np.zeros((8,5)) + np.nan
MeanLogTotalMedByAgeIncome = np.zeros((8,5)) + np.nan
StdevLogOOPmedByAgeIncome = np.zeros((8,5)) + np.nan
StdevLogTotalMedByAgeIncome = np.zeros((8,5)) + np.nan
InsuredRateByAgeIncome = np.zeros((8,5)) + np.nan
MeanPremiumByAgeIncome = np.zeros((8,5)) + np.nan
StdevPremiumByAgeIncome = np.zeros((8,5)) + np.nan
NoPremShareRateByAgeIncome = np.zeros((8,5)) + np.nan
OOPmomentWeightsByAgeIncome = np.zeros((8,5)) + np.nan
totMomentWeightsByAgeIncome = np.zeros((8,5)) + np.nan
premMomentWeightsByAgeIncome = np.zeros((8,5)) + np.nan
otherMomentWeightsByAgeIncome = np.zeros((8,5)) + np.nan
for j in range(40):
    i = int(raw_moments[j][1])-1
    k = int(raw_moments[j][0])-1
    MeanLogOOPmedByAgeIncome[i,k] = float(raw_moments[j][2])
    MeanLogTotalMedByAgeIncome[i,k] = float(raw_moments[j][3])
    StdevLogOOPmedByAgeIncome[i,k] = float(raw_moments[j][4])
    StdevLogTotalMedByAgeIncome[i,k] = float(raw_moments[j][5])
    InsuredRateByAgeIncome[i,k] = float(raw_moments[j][6])
    MeanPremiumByAgeIncome[i,k] = float(raw_moments[j][7])
    StdevPremiumByAgeIncome[i,k] = float(raw_moments[j][8])
    NoPremShareRateByAgeIncome[i,k] = float(raw_moments[j][9])
    OOPmomentWeightsByAgeIncome[i,k] = float(raw_moments[j][10])
    totMomentWeightsByAgeIncome[i,k] = float(raw_moments[j][11])
    premMomentWeightsByAgeIncome[i,k] = float(raw_moments[j][12])
    otherMomentWeightsByAgeIncome[i,k] = float(raw_moments[j][13])
    
    
# Load the moments by five-year age groups and health into a CSV reader object
data_location = os.path.dirname(os.path.abspath(__file__))
f = open(data_location + '\MomentsByAgeHealth.txt','r')
moment_reader = csv.reader(f,delimiter='\t')
raw_moments = list(moment_reader)
f.close()

# Store the moments by five-year age groups and health in arrays
MeanLogOOPmedByAgeHealth = np.zeros((12,5)) + np.nan
MeanLogTotalMedByAgeHealth = np.zeros((12,5)) + np.nan
StdevLogOOPmedByAgeHealth = np.zeros((12,5)) + np.nan
StdevLogTotalMedByAgeHealth = np.zeros((12,5)) + np.nan
InsuredRateByAgeHealth = np.zeros((8,5)) + np.nan
MeanPremiumByAgeHealth = np.zeros((8,5)) + np.nan
StdevPremiumByAgeHealth = np.zeros((8,5)) + np.nan
NoPremShareRateByAgeHealth = np.zeros((8,5)) + np.nan
OOPmomentWeightsByAgeHealth = np.zeros((12,5)) + np.nan
totMomentWeightsByAgeHealth = np.zeros((12,5)) + np.nan
for j in range(60):
    i = int(raw_moments[j][1])-1
    k = int(raw_moments[j][0])-1
    MeanLogOOPmedByAgeHealth[i,k] = float(raw_moments[j][2])
    MeanLogTotalMedByAgeHealth[i,k] = float(raw_moments[j][3])
    StdevLogOOPmedByAgeHealth[i,k] = float(raw_moments[j][4])
    StdevLogTotalMedByAgeHealth[i,k] = float(raw_moments[j][5])
    if i < 8:
        InsuredRateByAgeHealth[i,k] = float(raw_moments[j][6])
        MeanPremiumByAgeHealth[i,k] = float(raw_moments[j][7])
        StdevPremiumByAgeHealth[i,k] = float(raw_moments[j][8])
        NoPremShareRateByAgeHealth[i,k] = float(raw_moments[j][9])
        OOPmomentWeightsByAgeHealth[i,k] = float(raw_moments[j][10])
        totMomentWeightsByAgeHealth[i,k] = float(raw_moments[j][11])
    
    
# Load the moments for wealth-to-income ratio by age
data_location = os.path.dirname(os.path.abspath(__file__))
f = open(data_location + '\WealthByAge.txt','r')
moment_reader = csv.reader(f,delimiter='\t')
raw_moments = list(moment_reader)
f.close()

# Store the moments for wealth-to-income ratio by age
WealthRatioByAge = np.zeros(40) + np.nan
WealthMomentWeightsByAge = np.zeros(40) + np.nan
for j in range(40):
    WealthRatioByAge[j] = float(raw_moments[j][1])
    WealthMomentWeightsByAge[j] = float(raw_moments[j][2])


# Load the moments for wealth-to-income ratio by age and income quintile
data_location = os.path.dirname(os.path.abspath(__file__))
f = open(data_location + '\WealthByAgeIncome.txt','r')
moment_reader = csv.reader(f,delimiter='\t')
raw_moments = list(moment_reader)
f.close()

# Store the moments for wealth-to-income ratio by age
WealthRatioByAgeIncome = np.zeros((8,5)) + np.nan
WealthMomentWeightsByAgeIncome = np.zeros((8,5)) + np.nan
for j in range(40):
    i = int(raw_moments[j][1])-1
    k = int(raw_moments[j][0])-1
    WealthRatioByAgeIncome[i,k] = float(raw_moments[j][2])
    WealthMomentWeightsByAgeIncome[i,k] = float(raw_moments[j][3])
  
# Combine all data moments into a single 1D array
MomentList = [WealthRatioByAge,
              MeanLogTotalMedByAge,
              StdevLogTotalMedByAge,
              InsuredRateByAge,
              NoPremShareRateByAge,
              MeanPremiumByAge,
              StdevPremiumByAge,
              MeanLogTotalMedByAgeHealth.flatten(),
              StdevLogTotalMedByAgeHealth.flatten(),
              WealthRatioByAgeIncome.flatten(),
              MeanLogTotalMedByAgeIncome.flatten(),
              StdevLogTotalMedByAgeIncome.flatten(),
              InsuredRateByAgeIncome.flatten(),
              MeanPremiumByAgeIncome.flatten(),
              MeanLogOOPmedByAge.flatten(),
              StdevLogOOPmedByAge.flatten()]
data_moments = np.hstack(MomentList)

# Construct the vector of moment weights
if use_data_weights:
    moment_weights = np.zeros_like(data_moments) + np.nan
    moment_weights[0:40]    = WealthMomentWeightsByAge
    moment_weights[40:100]  = totMomentWeightsByAge
    moment_weights[100:160] = totMomentWeightsByAge
    moment_weights[160:200] = otherMomentWeightsByAge
    moment_weights[200:240] = premMomentWeightsByAge
    moment_weights[240:280] = premMomentWeightsByAge
    moment_weights[280:320] = premMomentWeightsByAge
    moment_weights[320:380] = totMomentWeightsByAgeHealth.flatten()
    moment_weights[380:440] = totMomentWeightsByAgeHealth.flatten()
    moment_weights[440:480] = WealthMomentWeightsByAgeIncome.flatten()
    moment_weights[480:520] = totMomentWeightsByAgeIncome.flatten()
    moment_weights[520:560] = totMomentWeightsByAgeIncome.flatten()
    moment_weights[560:600] = otherMomentWeightsByAgeIncome.flatten()
    moment_weights[600:640] = premMomentWeightsByAgeIncome.flatten()
    moment_weights[640:700] = OOPmomentWeightsByAge
    moment_weights[700:760] = OOPmomentWeightsByAge
    
else:
    moment_weights = np.ones_like(data_moments)

    # Make a moment weighting vector by turning on/off each type of moment
if not MomentBools[0]:
    moment_weights[0:40] = 0.0
if not MomentBools[1]:
    moment_weights[40:100] = 0.0
if not MomentBools[2]:
    moment_weights[100:160] = 0.0
if not MomentBools[3]:
    moment_weights[160:200] = 0.0
if not MomentBools[4]:
    moment_weights[200:240] = 0.0
if not MomentBools[5]:
    moment_weights[240:280] = 0.0
if not MomentBools[6]:
    moment_weights[280:320] = 0.0
if not MomentBools[7]:
    moment_weights[320:380] = 0.0
if not MomentBools[8]:
    moment_weights[380:440] = 0.0
if not MomentBools[9]:
    moment_weights[440:480] = 0.0
if not MomentBools[10]:
    moment_weights[480:520] = 0.0
if not MomentBools[11]:
    moment_weights[520:560] = 0.0
if not MomentBools[12]:
    moment_weights[560:600] = 0.0
if not MomentBools[13]:
    moment_weights[600:640] = 0.0
if not MomentBools[14]:
    moment_weights[640:700] = 0.0
if not MomentBools[15]:
    moment_weights[700:760] = 0.0

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    os.chdir('..')
    os.chdir('Figures')
    
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
    
    plt.plot(OneYearAge,InsuredRateByAge,'.k')
    plt.plot(FiveYearAge,InsuredRateByAgeIncome)
    plt.xlabel('Age')
    plt.ylabel('ESI uptake rate')
    plt.legend(['Overall average','Bottom income quintile','Second income quintile','Third income quintile','Fourth income quintile','Top income quintile'],loc=0,fontsize=8)
    plt.savefig('InsuredRateByAgeIncome.pdf')
    plt.show()
    
    plt.plot(OneYearAge,MeanPremiumByAge,'.k')
    plt.plot(FiveYearAge,MeanPremiumByAgeIncome)
    plt.xlabel('Age')
    plt.ylabel('Mean OOP premium')
    plt.ylim([500,3500])
    plt.legend(['Overall average','Bottom income quintile','Second income quintile','Third income quintile','Fourth income quintile','Top income quintile'],loc=0,fontsize=8)
    plt.savefig('MeanPremiumByAgeIncome.pdf')
    plt.show()
    
    plt.plot(OneYearAge,StdevPremiumByAge,'.k')
    plt.plot(FiveYearAge,StdevPremiumByAgeIncome)
    plt.xlabel('Age')
    plt.ylabel('Stdev OOP premium')
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
    
    plt.plot(OneYearAge,WealthRatioByAge,'.k')
    plt.plot(FiveYearAge,WealthRatioByAgeIncome)
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
    
    plt.plot(OneYearAge,InsuredRateByAge,'.k')
    plt.plot(FiveYearAge,InsuredRateByAgeHealth)
    plt.xlabel('Age')
    plt.ylabel('ESI uptake rate')
    plt.legend(['Overall average','Poor health','Fair health','Good health','Very good health','Excellent health'],loc=0,fontsize=8)
    plt.savefig('InsuredRateByAgeHealth.pdf')
    plt.show()
    
    plt.plot(OneYearAge,MeanPremiumByAge,'.k')
    plt.plot(FiveYearAge,MeanPremiumByAgeHealth)
    plt.xlabel('Age')
    plt.ylabel('Mean OOP premium')
    plt.ylim([500,3500])
    plt.legend(['Overall average','Poor health','Fair health','Good health','Very good health','Excellent health'],loc=0,fontsize=8)
    plt.savefig('MeanPremiumByAgeHealth.pdf')
    plt.show()
    
    plt.plot(OneYearAge,StdevPremiumByAge,'.k')
    plt.plot(FiveYearAge,StdevPremiumByAgeHealth)
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

