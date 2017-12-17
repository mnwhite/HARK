'''
This module contains calibrated / pre-estimated / non-structural parameters for
the health investment model, as well as test values of the structural parameters.
'''

from copy import copy
import numpy as np

# Choose state grid sizes and bounds (exogenously chosen)
Hcount = 16
aXtraCount = 48
hCount = 2*(Hcount-1)+1
bNrmCount = 2*aXtraCount
MedShkCount = 24
aXtraMin = 0.001
aXtraMax = 200
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

# Make a dictionary with insurance function parameters (copied from paper table)
insurance_params = {
    'Premium0' : -0.0058,
    'PremiumHealth' : 0.0803,
    'PremiumHealthSq' : 0.0130,
    'PremiumAge' : .0036,
    'PremiumAgeSq' : -0.00008,
    'PremiumSex' : -0.0088,
    'PremiumInc' : 0.0216,
    'PremiumIncSq' : -0.0001,
    'PremiumIncCu' : 0.00000000673,
    'PremiumHealthAge' : 0.0039,
    'PremiumHealthSqAge' : -0.0068,
    'PremiumHealthAgeSq' : 0.000034,
    'PremiumHealthSqAgeSq' : -0.000024,
    'PremiumHealthInc' : -0.0324,
    'PremiumHealthSqInc' : 0.0133,
    'PremiumHealthIncSq' : 0.00007,
    'PremiumHealthSqIncSq' : 0.000032,
    'Copay0' : 0.0436,
    'CopayHealth' : 0.4325,
    'CopayHealthSq' : -0.1505,
    'CopayAge' : .0034,
    'CopayAgeSq' : -0.000013,
    'CopaySex' : -0.0409,
    'CopayInc' : 0.0337,
    'CopayIncSq' : -0.0012,
    'CopayIncCu' : 0.000000606,
    'CopayHealthAge' : -0.0067,
    'CopayHealthSqAge' : 0.0032,
    'CopayHealthAgeSq' : 0.00000438,
    'CopayHealthSqAgeSq' : -0.00012,
    'CopayHealthInc' : -0.0539,
    'CopayHealthSqInc' : 0.0268,
    'CopayHealthIncSq' : 0.0024,
    'CopayHealthSqIncSq' : -0.00136,
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
    'Sex' : 0.0,
    'cycles' : 1,
    'DataToSimRepFactor' : 100
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

# Make a test parameter vector
test_param_vec = np.array([
    0.986282156309,     # 0 CRRA
    0.889145326143,     # 1 DiscFac
    1.74000477361,      # 2 MedCurve
    -0.902113128594,    # 3 LifeUtility
    -0.5,               # 4 MargUtilityShift
    0.136535220024,     # 5 Cfloor
    7.49251475136,      # 6 Bequest0
    2.29160750023,      # 7 Bequest1
    0.0718333405867,    # 8 MedShkMean0
    0.196233326869,     # 9 MedShkMeanSex
    0.287511255612,     # 10 MedShkMeanAge
    -0.00905188245257,  # 11 MedShkMeanAgeSq
    -7.67558372995,     # 12 MedShkMeanHealth
    1.77510262466,      # 13 MedShkMeanHealthSq
    2.71593990024,      # 14 MedShkStd0
    -0.0603166861082,   # 15 MedShkStd1
    0.0163381242443,    # 16 HealthNext0
    -0.00396380623316,  # 17 HealthNextSex
    -0.000638834972427, # 18 HealthNextAge
    -0.000290561156297, # 19 HealthNextAgeSq
    0.873979878446,     # 20 HealthNextHealth
    0.0806898912141,    # 21 HealthNextHealthSq
    0.170054257328,     # 22 HealthShkStd0
    -0.0791852964486,   # 23 HealthShkStd1
    -4.70222862043,     # 24 HealthProd0 (in log)
    -np.inf,            # 25 HealthProd1 (in log)
    0.000,              # 26 HealthProd2
    -0.49554640989,     # 27 Mortality0
    0.338606078662,     # 28 MortalitySex
    -0.00189140591818,  # 29 MortalityAge
    0.00614760053852,   # 30 MortalityAgeSq
    -2.61134057329,     # 31 MortalityHealth
    0.730883566197      # 32 MortalityHealthSq
    ])

#   1.0030272606,       # 0 CRRA

#    2.0,                # 0 CRRA
#    0.826851467639,     # 1 DiscFac
#    1.61467857653,      # 2 MedCurve
#    0.0978868714061-1.0,# 3 LifeUtility
#    -0.5,               # 4 MargUtilityShift
#    0.371107323164,     # 5 Cfloor
#    0.492413649548,     # 6 Bequest0
#    1.57686798194,      # 7 Bequest1

#    -3.09286007811,     # 25 HealthProd1 (in log)

# These are only used by the estimation to decide when to write parameters to disk
func_call_count = 0
store_freq = 50

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
    'HealthProd0',
    'HealthProd1',
    'HealthProd2',
    'Mortality0',
    'MortalitySex',
    'MortalityAge',
    'MortalityAgeSq',
    'MortalityHealth',
    'MortalityHealthSq'
    ]