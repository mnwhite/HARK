'''
This module contains calibrated / pre-estimated / non-structural parameters for
the health investment model, as well as test values of the structural parameters.
'''

from copy import copy
import numpy as np
import csv
from MakeTables import makeInsuranceTable, makeHealthProbitTable

# Choose state grid sizes and bounds (exogenously chosen)
Hcount = 16
aXtraCount = 48
hCount = 2*(Hcount-1)+1
bNrmCount = 2*aXtraCount
MedShkCount = 24
aXtraMin = 0.001
aXtraMax = 100
aXtraNestFac = 3
T_cycle = 25

# Make a dictionary of grid sizes
grid_size_params = {
    'Hcount' : Hcount,
    'aXtraCount' : aXtraCount,
    'hCount' : hCount,
    'bNrmCount' : bNrmCount,
    'MedShkCount' : MedShkCount,
    'aXtraMin' : aXtraMin,
    'aXtraMax' : aXtraMax,
    'aXtraNestFac' : aXtraNestFac,
    'aXtraExtra' : [None],
    'T_cycle' : T_cycle
}

# Load the health probit coefficients into memory
infile = open('./Data/HealthCoeffs.txt','r') 
my_reader = csv.reader(infile,delimiter=',')
health_coeffs_raw = list(my_reader)[1]
infile.close()
health_coeffs = np.zeros(50)
health_stderrs = np.zeros(50)
for i in range(50):
    health_coeffs[i] = float(health_coeffs_raw[i])
    health_stderrs[i] = float(health_coeffs_raw[i+50])
makeHealthProbitTable(health_coeffs,health_stderrs,'HealthProbit')


# Load the insurance coefficients into memory
infile = open('./Data/InsuranceCoeffs.txt','r') 
my_reader = csv.reader(infile,delimiter=',')
insurance_coeffs_raw = list(my_reader)[1]
infile.close()
copay_coeffs = np.zeros(17)
premium_coeffs = np.zeros(17)
copay_stderrs = np.zeros(17)
premium_stderrs = np.zeros(17)
for i in range(17):
    copay_coeffs[i] = float(insurance_coeffs_raw[i])
    premium_coeffs[i] = float(insurance_coeffs_raw[i+17])
    copay_stderrs[i] = float(insurance_coeffs_raw[i+34])
    premium_stderrs[i] = float(insurance_coeffs_raw[i+51])
    
# Make an insurance table for the paper
makeInsuranceTable(copay_coeffs,premium_coeffs,copay_stderrs,premium_stderrs,'Insurance')
    
# Make a dictionary with insurance function parameters (loaded from Stata output)
insurance_params = {
    'PremiumHealth' : premium_coeffs[0],
    'PremiumHealthSq' : premium_coeffs[1],
    'PremiumAge' : premium_coeffs[2],
    'PremiumAgeSq' : premium_coeffs[3],
    'PremiumSex' : premium_coeffs[4],
    'PremiumInc' : premium_coeffs[5],
    'PremiumIncSq' : premium_coeffs[6],
    'PremiumIncCu' : premium_coeffs[7],
    'PremiumHealthAge' : premium_coeffs[8],
    'PremiumHealthSqAge' : premium_coeffs[9],
    'PremiumHealthAgeSq' : premium_coeffs[10],
    'PremiumHealthSqAgeSq' : premium_coeffs[11],
    'PremiumHealthInc' : premium_coeffs[12],
    'PremiumHealthSqInc' : premium_coeffs[13],
    'PremiumHealthIncSq' : premium_coeffs[14],
    'PremiumHealthSqIncSq' : premium_coeffs[15],
    'Premium0' : premium_coeffs[16],
    'CopayHealth' : copay_coeffs[0],
    'CopayHealthSq' : copay_coeffs[1],
    'CopayAge' : copay_coeffs[2],
    'CopayAgeSq' : copay_coeffs[3],
    'CopaySex' : copay_coeffs[4],
    'CopayInc' : copay_coeffs[5],
    'CopayIncSq' : copay_coeffs[6],
    'CopayIncCu' : copay_coeffs[7],
    'CopayHealthAge' : copay_coeffs[8],
    'CopayHealthSqAge' : copay_coeffs[9],
    'CopayHealthAgeSq' : copay_coeffs[10],
    'CopayHealthSqAgeSq' : copay_coeffs[11],
    'CopayHealthInc' : copay_coeffs[12],
    'CopayHealthSqInc' : copay_coeffs[13],
    'CopayHealthIncSq' : copay_coeffs[14],
    'CopayHealthSqIncSq' : copay_coeffs[15],
    'Copay0' : copay_coeffs[16],
}


# Make a dictionary with example basic parameters
other_exog_params = {
    'Income0' : 8.0,
    'IncomeAge' : 0.0,
    'IncomeAgeSq' : 0.0,
    'IncomeAgeCu' : 0.0,
    'IncomeAgeQu' : 0.0,
    'MedPrice0' : 1.0,
    'Rfree' : 1.04,
    'Subsidy0' : 0.0,
    'Subsidy1' : 0.0,
    'CalcExpectationFuncs' : False,
    'CalcSocialOptimum' : False,
    'SameCopayForMedAndInvst' : True,
    'DeleteSolution' : True,
    'LifePrice' : 0.,
    'Sex' : 0.0,
    'cycles' : 1,
    'T_sim' : 25,
    'DataToSimRepFactor' : 50
}

# Make a dictionary with structural parameters for testing
struct_test_params = {
    'CRRA' : 2.0,
    'DiscFac' : 0.90,
    'MedCurve' : 6.0,
    'LifeUtility' : 0.8,
    'MargUtilityShift' : -0.5,
    'Cfloor' : 0.8,
    'Bequest0' : 1.0,
    'Bequest1' : 4.0,
    'MedShkMean0' : -0.65,
    'MedShkMeanSex' : -0.5,
    'MedShkMeanAge' : 0.11,
    'MedShkMeanAgeSq' : 0.002,
    'MedShkMeanHealth' : -1.0,
    'MedShkMeanHealthSq' : -3.0,
    'MedShkStd0' : 2.0,
    'MedShkStd1' : -0.2,
    'HealthNext0' : 0.03,
    'HealthNextSex' : -0.005,
    'HealthNextAge' : -0.003,
    'HealthNextAgeSq' : -0.000006,
    'HealthNextHealth' : 0.82,
    'HealthNextHealthSq' : 0.159,
    'HealthShkStd0' : 0.13,
    'HealthShkStd1' : -0.04,
    'HealthProd0' : 0.00,
    'HealthProd1' : 0.000,
    'HealthProd2' : 0.000,
    'Mortality0' : -0.79,
    'MortalitySex' : 0.29,
    'MortalityAge' : -0.01,
    'MortalityAgeSq' : 0.007,
    'MortalityHealth' : -1.5,
    'MortalityHealthSq' : -0.7
}

# Make a test dictionary
test_params = copy(struct_test_params)
test_params.update(other_exog_params)
test_params.update(insurance_params)
test_params.update(grid_size_params)

# Make a dictionary of basic exogenous parameters
basic_estimation_dict = copy(other_exog_params)
basic_estimation_dict.update(insurance_params)
basic_estimation_dict.update(grid_size_params)

# These are the estimated parameters in the full model
full_model_param_vec = np.array([
    0.395747637459,      # 0 CRRAcon
    0.954441535586,      # 1 DiscFac
    2.74409486877,       # 2 CRRAmed
    2.17014226367,       # 3 LifeUtility
    0.0,                 # 4 MargUtilityShift
    1.04831193007,       # 5 Cfloor
    11.0737301651,       # 6 Bequest0
    1.80315880232,       # 7 Bequest1
    -2.32454945012,      # 8 MedShkMean0
    -0.713908448846,     # 9 MedShkMeanSex
    0.44600307886,       # 10 MedShkMeanAge
    -0.015201358284,     # 11 MedShkMeanAgeSq
    -8.3210616052,       # 12 MedShkMeanHealth
    -0.0115458059298,    # 13 MedShkMeanHealthSq
    2.73086670706,       # 14 MedShkStd0
    0.373632264832,      # 15 MedShkStd1
    0.0656833979957,     # 16 HealthNext0
    -0.00696450550332,   # 17 HealthNextSex
    -0.00022776292138,   # 18 HealthNextAge
    -0.000325984513277,  # 19 HealthNextAgeSq
    0.664388775961,      # 20 HealthNextHealth
    0.243683208401,      # 21 HealthNextHealthSq
    0.172382175469,      # 22 HealthShkStd0
    -0.0894647939341,    # 23 HealthShkStd1
    15.5611402001,       # 24 LogJerk
    -2.13369276099,      # 25 LogSlope
    1.71842956397,       # 26 LogCurve
    -0.488972328165,     # 27 Mortality0
    0.327125262504,      # 28 MortalitySex
    -7.4458983434e-05,   # 29 MortalityAge
     0.0058979528662,    # 30 MortalityAgeSq
    -2.23644923978,      # 31 MortalityHealth
    0.0359369543779,     # 32 MortalityHealthSq
    ])
 
# These are the estimated parameters in the no health investment model
no_investment_param_vec = np.array([
    0.791015158188,      # 0 CRRAcon
    0.962706974916,      # 1 DiscFac
    1.5675322475,        # 2 CRRAmed
    2.17014226367,       # 3 LifeUtility
    0.0,                 # 4 MargUtilityShift
    1.18421039632,       # 5 Cfloor
    8.7463860231,        # 6 Bequest0
    2.44900118467,       # 7 Bequest1
    -5.78274381076,      # 8 MedShkMean0
    -0.962334897288,     # 9 MedShkMeanSex
    0.306633415549,      # 10 MedShkMeanAge
    -0.00164869717987,   # 11 MedShkMeanAgeSq
    -6.40314402034,      # 12 MedShkMeanHealth
    -0.0799909164657,    # 13 MedShkMeanHealthSq
    4.63551568651,       # 14 MedShkStd0
    -1.5732557546,       # 15 MedShkStd1
    0.0155042533703,     # 16 HealthNext0
    -0.0038384135423,    # 17 HealthNextSex
    -0.000235291171021,  # 18 HealthNextAge
    -0.000335534473032,  # 19 HealthNextAgeSq
    0.843531662843,      # 20 HealthNextHealth
    0.120833457634,      # 21 HealthNextHealthSq
    0.183501817828,      # 22 HealthShkStd0
    -0.105368198316,     # 23 HealthShkStd1
    15.5611402001,       # 24 LogJerk
    -np.inf,             # 25 LogSlope
    1.71842956397,       # 26 LogCurve
    -0.460781109508,     # 27 Mortality0
    0.333732265468,      # 28 MortalitySex
    -0.001818754439,     # 29 MortalityAge
     0.00619148618625,   # 30 MortalityAgeSq
    -2.76138537804,      # 31 MortalityHealth
    0.788656291627,      # 32 MortalityHealthSq
    ])
    
test_param_vec = full_model_param_vec
    


# These are only used by the estimation to decide when to write parameters to disk
func_call_count = 0
store_freq = 5

# Make a list of parameter names corresponding to their position in the vector above
param_names = [
    'CRRA',
    'DiscFac',
    'MedCurve',
    'LifeUtility',
    'MargUtilityShift',
    'Cfloor',
    'Bequest0',
    'Bequest1',
    'MedShkMean0',
    'MedShkMeanSex',
    'MedShkMeanAge',
    'MedShkMeanAgeSq',
    'MedShkMeanHealth',
    'MedShkMeanHealthSq',
    'MedShkStd0',
    'MedShkStd1',
    'HealthNext0',
    'HealthNextSex',
    'HealthNextAge',
    'HealthNextAgeSq',
    'HealthNextHealth',
    'HealthNextHealthSq',
    'HealthShkStd0',
    'HealthShkStd1',
    'LogJerk',
    'LogSlope',
    'LogCurve',
    'Mortality0',
    'MortalitySex',
    'MortalityAge',
    'MortalityAgeSq',
    'MortalityHealth',
    'MortalityHealthSq'
    ]