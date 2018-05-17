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
    'Subsidy0' : 0.0,
    'Subsidy1' : 0.0,
    'CalcExpectationFuncs' : False,
    'CalcSocialOptimum' : False,
    'Sex' : 0.0,
    'cycles' : 1,
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

# Make a test parameter vector
test_param_vec = np.array([
    1.31227464968,      # 0 CRRA
    0.951743507223,     # 1 DiscFac
    5.92045230814,      # 2 MedCurve
    2.33386438153,      # 3 LifeUtility
    0.0,                # 4 MargUtilityShift
    0.222906767166,     # 5 Cfloor
    11.4460359759,      # 6 Bequest0
    3.80938776521,      # 7 Bequest1
    -1.27333768705,     # 8 MedShkMean0
    0.0,                # 9 MedShkMeanSex
    0.221726768121,     # 10 MedShkMeanAge
    -0.00286998248716,  # 11 MedShkMeanAgeSq
    -6.67629226678,     # 12 MedShkMeanHealth
    -0.0615004369575,   # 13 MedShkMeanHealthSq
    1.98861616794,      # 14 MedShkStd0
    0.303606509789,     # 15 MedShkStd1
    -0.0011637681873,   # 16 HealthNext0
    -0.00805163660507,  # 17 HealthNextSex
    -0.000819783048845, # 18 HealthNextAge
    -0.00034825565717,  # 19 HealthNextAgeSq
    0.87996678252,      # 20 HealthNextHealth
    0.0781600175873,    # 21 HealthNextHealthSq
    0.17037348533,      # 22 HealthShkStd0
    -0.0804025772933,   # 23 HealthShkStd1
    -14.8839894417,     # 24 LogJerk
    -1.53126486432,     # 25 LogSlope
    2.06452653773,      # 26 LogCurve
    -0.396828479727,    # 27 Mortality0
    0.342775636977,     # 28 MortalitySex
    -0.000622613919962, # 29 MortalityAge
    0.0062259993173,    # 30 MortalityAgeSq
    -3.20969450762,     # 31 MortalityHealth
    1.33098832529,      # 32 MortalityHealthSq
    ])

#    1.65857373243,      # 3 LifeUtility
#    -73.8460301235,     # 24 HealthProd0
#    -1.92191203403,     # 25 HealthProd1 (log initial slope)
#    1.67989170316,      # 26 HealthProd2 (log initial curvature)

#    -16.0,              # 24 LogJerk
#    -1.92191203403,     # 25 LogSlope
#    1.67989170316,      # 26 LogCurve



# These are only used by the estimation to decide when to write parameters to disk
func_call_count = 0
store_freq = 10

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