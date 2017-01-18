'''
This module unpacks data moments as calculated in the MEPS, HRS, and SCF, loading them as arrays.
'''
import numpy as np
import csv
import os
make_figs = True

# Load the moments by one-year age groups into a CSV reader object
data_location = os.path.dirname(os.path.abspath(__file__))
f = open(data_location + '\MomentsByAge.txt','r')
moment_reader = csv.reader(f,delimiter='\t')
raw_moments = list(moment_reader)
f.close()

# Store the moments by one-year age groups in arrays
OneYearAge = np.arange(25,65)
MeanLogOOPmedByAge = np.zeros(40) + np.nan
MeanLogTotalMedByAge = np.zeros(40) + np.nan
StdevLogOOPmedByAge = np.zeros(40) + np.nan
StdevLogTotalMedByAge = np.zeros(40) + np.nan
InsuredRateByAge = np.zeros(40) + np.nan
MeanPremiumByAge = np.zeros(40) + np.nan
StdevPremiumByAge = np.zeros(40) + np.nan
NoPremShareRateByAge = np.zeros(40) + np.nan
for j in range(40):
    MeanLogOOPmedByAge[j] = float(raw_moments[j][1])
    MeanLogTotalMedByAge[j] = float(raw_moments[j][2])
    StdevLogOOPmedByAge[j] = float(raw_moments[j][3])
    StdevLogTotalMedByAge[j] = float(raw_moments[j][4])
    InsuredRateByAge[j] = float(raw_moments[j][5])
    MeanPremiumByAge[j] = float(raw_moments[j][6])
    StdevPremiumByAge[j] = float(raw_moments[j][7])
    NoPremShareRateByAge[j] = float(raw_moments[j][8])
    

# Load the moments by five-year age groups and income quintile into a CSV reader object
data_location = os.path.dirname(os.path.abspath(__file__))
f = open(data_location + '\MomentsByAgeIncome.txt','r')
moment_reader = csv.reader(f,delimiter='\t')
raw_moments = list(moment_reader)
f.close()

# Store the moments by five-year age groups and income quintile in arrays
FiveYearAge = 5*np.arange(8) + 27
MeanLogOOPmedByAgeIncome = np.zeros((8,5)) + np.nan
MeanLogTotalMedByAgeIncome = np.zeros((8,5)) + np.nan
StdevLogOOPmedByAgeIncome = np.zeros((8,5)) + np.nan
StdevLogTotalMedByAgeIncome = np.zeros((8,5)) + np.nan
InsuredRateByAgeIncome = np.zeros((8,5)) + np.nan
MeanPremiumByAgeIncome = np.zeros((8,5)) + np.nan
StdevPremiumByAgeIncome = np.zeros((8,5)) + np.nan
NoPremShareRateByAgeIncome = np.zeros((8,5)) + np.nan
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
    
    
# Load the moments by five-year age groups and health into a CSV reader object
data_location = os.path.dirname(os.path.abspath(__file__))
f = open(data_location + '\MomentsByAgeHealth.txt','r')
moment_reader = csv.reader(f,delimiter='\t')
raw_moments = list(moment_reader)
f.close()

# Store the moments by five-year age groups and health in arrays
MeanLogOOPmedByAgeHealth = np.zeros((8,5)) + np.nan
MeanLogTotalMedByAgeHealth = np.zeros((8,5)) + np.nan
StdevLogOOPmedByAgeHealth = np.zeros((8,5)) + np.nan
StdevLogTotalMedByAgeHealth = np.zeros((8,5)) + np.nan
InsuredRateByAgeHealth = np.zeros((8,5)) + np.nan
MeanPremiumByAgeHealth = np.zeros((8,5)) + np.nan
StdevPremiumByAgeHealth = np.zeros((8,5)) + np.nan
NoPremShareRateByAgeHealth = np.zeros((8,5)) + np.nan
for j in range(40):
    i = int(raw_moments[j][0])-1
    k = int(raw_moments[j][1])-1
    MeanLogOOPmedByAgeHealth[i,k] = float(raw_moments[j][2])
    MeanLogTotalMedByAgeHealth[i,k] = float(raw_moments[j][3])
    StdevLogOOPmedByAgeHealth[i,k] = float(raw_moments[j][4])
    StdevLogTotalMedByAgeHealth[i,k] = float(raw_moments[j][5])
    InsuredRateByAgeHealth[i,k] = float(raw_moments[j][6])
    MeanPremiumByAgeHealth[i,k] = float(raw_moments[j][7])
    StdevPremiumByAgeHealth[i,k] = float(raw_moments[j][8])
    NoPremShareRateByAgeHealth[i,k] = float(raw_moments[j][9])
    
    
# Load the moments for wealth-to-income ratio by age
data_location = os.path.dirname(os.path.abspath(__file__))
f = open(data_location + '\WealthByAge.txt','r')
moment_reader = csv.reader(f,delimiter='\t')
raw_moments = list(moment_reader)
f.close()

# Store the moments for wealth-to-income ratio by age
WealthRatioByAge = np.zeros(40) + np.nan
for j in range(40):
    WealthRatioByAge[j] = float(raw_moments[j][1])


# Load the moments for wealth-to-income ratio by age and income quintile
data_location = os.path.dirname(os.path.abspath(__file__))
f = open(data_location + '\WealthByAgeIncome.txt','r')
moment_reader = csv.reader(f,delimiter='\t')
raw_moments = list(moment_reader)
f.close()

# Store the moments for wealth-to-income ratio by age
WealthRatioByAgeIncome = np.zeros((8,5)) + np.nan
for j in range(40):
    i = int(raw_moments[j][1])-1
    k = int(raw_moments[j][0])-1
    WealthRatioByAgeIncome[i,k] = float(raw_moments[j][2])   


if make_figs:
    import matplotlib.pyplot as plt
    os.chdir('..')
    os.chdir('Figures')
    
    plt.plot(OneYearAge,MeanLogOOPmedByAge,'.k')
    plt.plot(FiveYearAge,MeanLogOOPmedByAgeIncome)
    plt.xlabel('Age')
    plt.ylabel('Mean log OOP medical expenses')
    plt.legend(['Overall average','Bottom income quintile','Second income quintile','Third income quintile','Fourth income quintile','Top income quintile'],loc=0,fontsize=8)
    plt.savefig('MeanLogOOPmedByAgeIncome.pdf')
    plt.show()
    
    plt.plot(OneYearAge,MeanLogTotalMedByAge,'.k')
    plt.plot(FiveYearAge,MeanLogTotalMedByAgeIncome)
    plt.xlabel('Age')
    plt.ylabel('Mean log total medical expenses')
    plt.legend(['Overall average','Bottom income quintile','Second income quintile','Third income quintile','Fourth income quintile','Top income quintile'],loc=0,fontsize=8)
    plt.savefig('MeanLogTotalMedByAgeIncome.pdf')
    plt.show()
    
    plt.plot(OneYearAge,StdevLogOOPmedByAge,'.k')
    plt.plot(FiveYearAge,StdevLogOOPmedByAgeIncome)
    plt.xlabel('Age')
    plt.ylabel('Stdev log OOP medical expenses')
    plt.legend(['Overall average','Bottom income quintile','Second income quintile','Third income quintile','Fourth income quintile','Top income quintile'],loc=0,fontsize=8)
    plt.savefig('StdevLogOOPmedByAgeIncome.pdf')
    plt.show()
    
    plt.plot(OneYearAge,StdevLogTotalMedByAge,'.k')
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
    
    plt.plot(OneYearAge,MeanLogOOPmedByAge,'.k')
    plt.plot(FiveYearAge,MeanLogOOPmedByAgeHealth)
    plt.xlabel('Age')
    plt.ylabel('Mean log OOP medical expenses')
    plt.legend(['Overall average','Poor health','Fair health','Good health','Very good health','Excellent health'],loc=0,fontsize=8)
    plt.savefig('MeanLogOOPmedByAgeHealth.pdf')
    plt.show()
    
    plt.plot(OneYearAge,MeanLogTotalMedByAge,'.k')
    plt.plot(FiveYearAge,MeanLogTotalMedByAgeHealth)
    plt.xlabel('Age')
    plt.ylabel('Mean log total medical expenses')
    plt.legend(['Overall average','Poor health','Fair health','Good health','Very good health','Excellent health'],loc=0,fontsize=8)
    plt.savefig('MeanLogTotalMedByAgeHealth.pdf')
    plt.show()
    
    plt.plot(OneYearAge,StdevLogOOPmedByAge,'.k')
    plt.plot(FiveYearAge,StdevLogOOPmedByAgeHealth)
    plt.xlabel('Age')
    plt.ylabel('Stdev log OOP medical expenses')
    plt.legend(['Overall average','Poor health','Fair health','Good health','Very good health','Excellent health'],loc=0,fontsize=8)
    plt.savefig('StdevLogOOPmedByAgeHealth.pdf')
    plt.show()
    
    plt.plot(OneYearAge,StdevLogTotalMedByAge,'.k')
    plt.plot(FiveYearAge,StdevLogTotalMedByAgeHealth)
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

