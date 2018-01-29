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
    0.731286207828,     # 0 CRRA
    0.928694546256,     # 1 DiscFac
    2.06356193023,      # 2 MedCurve
    1.6323315393,       # 3 LifeUtility
    -0.5,               # 4 MargUtilityShift
    0.130429240181,     # 5 Cfloor
    10.2507822365,      # 6 Bequest0
    2.56988190806,      # 7 Bequest1
    0.268850369098,     # 8 MedShkMean0
    0.165894639552,     # 9 MedShkMeanSex
    0.361958865002,     # 10 MedShkMeanAge
    -0.00929330744163,  # 11 MedShkMeanAgeSq
    -10.3848870299,     # 12 MedShkMeanHealth
    1.26316856269,      # 13 MedShkMeanHealthSq
    2.02166951639,      # 14 MedShkStd0
    -0.0632194977819,   # 15 MedShkStd1
    -0.00192888449907,  # 16 HealthNext0
    -0.00606092503783,  # 17 HealthNextSex
    -0.000845692237038, # 18 HealthNextAge
    -0.000293401480164, # 19 HealthNextAgeSq
    0.867505428068,     # 20 HealthNextHealth
    0.0914703487628,    # 21 HealthNextHealthSq
    0.169998501912,     # 22 HealthShkStd0
    -0.0786653746195,   # 23 HealthShkStd1
    -17.2980078815,     # 24 LogJerk
    -1.61894432908,     # 25 LogSlope
    2.10032808271,      # 26 LogCurve
    -0.373553729449,    # 27 Mortality0
    0.359503071114,     # 28 MortalitySex
    -0.0026169085913,   # 29 MortalityAge
    0.00628738029294,   # 30 MortalityAgeSq
    -3.27529906734,     # 31 MortalityHealth
    1.38040276975,      # 32 MortalityHealthSq
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