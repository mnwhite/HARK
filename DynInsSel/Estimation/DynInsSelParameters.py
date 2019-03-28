'''
This module makes exogenous agent parameters for the DynInsSel project.
'''

from copy import copy
import numpy as np
from scipy.stats import norm
from scipy.optimize import newton
import matplotlib.pyplot as plt
import os
import csv
from HARKinterpolation import LinearInterp

# Set parameters for estimation
AgentCountTotal = 100000
StaticBool = False

# Calibrated / other parameters (grid sizes, etc)
Rfree = 5*[1.03]                    # Interest factor on assets
aXtraMin = 0.001                    # Minimum end-of-period "assets above minimum" value
aXtraMax = 16                       # Minimum end-of-period "assets above minimum" value               
aXtraExtra = [0.005,0.01]           # Some other value of "assets above minimum" to add to the grid, not used
aXtraNestFac = 1                    # Exponential nesting factor when constructing "assets above minimum" grid
aXtraCount = 32                     # Number of points in the grid of "assets above minimum"
PermShkCount = 5                    # Number of points in discrete approximation to permanent income shocks
TranShkCount = 5                    # Number of points in discrete approximation to transitory income shocks
UnempPrb = 0.05                     # Probability of unemployment while working
UnempPrbRet = 0.0000                # Probability of "unemployment" while retired
IncUnemp = 0.3                      # Unemployment benefits replacement rate
IncUnempRet = 0.0                   # "Unemployment" benefits when retired
BoroCnstArt = 0.0                   # Artificial borrowing constraint; imposed minimum level of end-of period assets
CubicBool = False                   # Use cubic spline interpolation when True, linear interpolation when False
DecurveBool = True                  # "Decurve" value through the inverse utility function when applying preference shocks when True
pLvlPctiles = np.concatenate(([0.001, 0.005, 0.01, 0.03], np.linspace(0.05, 0.95, num=12),[0.97, 0.99, 0.995, 0.999]))
pLvlInitStd = 0.4                   # Initial standard deviation of (log) permanent income
pLvlInitMean = 0.0                  # Initial average of log permanent income
PermIncCorr = 1.0                   # Serial correlation coefficient for permanent income
MedShkCount = 25                    # Number of medical shock points
DevMin = -3.0                       # Minimum standard deviations below MedShk mean
DevMax = 5.0                        # Maximum standard deviations above MedShk mean
MedPrice = 1.0                      # Relative price of a unit of medical care
AgentCount = 10000                  # Number of agents of this type (only matters for simulation)
DeductibleList = [0.06,0.05,0.04,0.03,0.02] # List of deductibles for working-age insurance contracts
T_sim = 60                          # Number of periods to simulate (age 25 to 84)


# Make a trivial array of transition probabilities among ESI states.
# Order: No ESI, ESI with no emp contribution, ESI with emp contribution
ESImrkvArray = np.array([[0.8,0.1,0.1],
                         [0.03,0.80,0.17],
                         [0.01,0.015,0.975]])
    
# Make survival probabilities by health state and age based on a probit on HRS data for ages 50-119
AgeMortParams = [-2.757652,.044746,-.0010514,.0000312,-1.62e-07]  # MEN ONLY
HealthMortAdj = [.0930001,.2299474,.3242607,.5351414]             # MEN ONLY

HealthMortCum = np.cumsum(HealthMortAdj)
MortProbitEX = lambda x : AgeMortParams[0] + AgeMortParams[1]*x + AgeMortParams[2]*x**2  + AgeMortParams[3]*x**3 + AgeMortParams[4]*x**4
MortProbitVG = lambda x : MortProbitEX(x) + HealthMortCum[0]
MortProbitGD = lambda x : MortProbitEX(x) + HealthMortCum[1]
MortProbitFR = lambda x : MortProbitEX(x) + HealthMortCum[2]
MortProbitPR = lambda x : MortProbitEX(x) + HealthMortCum[3]
LivPrbOld = np.zeros((5,70))
Age = np.arange(70)
BiannualTransform = lambda f, x : (1.0 - norm.cdf(f(x)))**0.5
LivPrbOld[0,:] = BiannualTransform(MortProbitPR,Age)
LivPrbOld[1,:] = BiannualTransform(MortProbitFR,Age)
LivPrbOld[2,:] = BiannualTransform(MortProbitGD,Age)
LivPrbOld[3,:] = BiannualTransform(MortProbitVG,Age)
LivPrbOld[4,:] = BiannualTransform(MortProbitEX,Age)

# Import survival probabilities from SSA data, from age 0 to age 119
data_location = os.path.dirname(os.path.abspath(__file__))
f = open(data_location + '/' + 'USactuarial.txt','r')
actuarial_reader = csv.reader(f,delimiter='\t')
raw_actuarial = list(actuarial_reader)
DiePrb = []
for j in range(len(raw_actuarial)):
    DiePrb += [float(raw_actuarial[j][1])] # This uses male death probabilities
f.close
DiePrb = np.array(DiePrb)[24:] # Take from age 24 onward
DiePrbBiannual = 1.0 - (1.0 - DiePrb[:-1])*(1.0 - DiePrb[1:])

# Specify the initial distribution of health at age 24-26, taken directly from MEPS data
HealthPrbsInit = [0.003,0.036,0.196,0.348,0.417]
HealthPrbsInit_d = np.array([0.004,0.063,0.304,0.297,0.332])
HealthPrbsInit_h = np.array([0.004,0.042,0.202,0.346,0.406])
HealthPrbsInit_c = np.array([0.003,0.019,0.163,0.363,0.452])
MrkvPrbsInit_d = np.concatenate([0.3*HealthPrbsInit_d,0.05*HealthPrbsInit_d,0.65*HealthPrbsInit_d])
MrkvPrbsInit_h = np.concatenate([0.2*HealthPrbsInit_h,0.05*HealthPrbsInit_h,0.75*HealthPrbsInit_h])
MrkvPrbsInit_c = np.concatenate([0.1*HealthPrbsInit_c,0.05*HealthPrbsInit_c,0.85*HealthPrbsInit_c])
EducWeight = [0.080,0.566,0.354]

# Make the income shock standard deviations by age, from age 25-120
# These might need revising
retired_T = 55
working_T = 40
AgeCount = retired_T + working_T
T_cycle = retired_T + working_T
TranShkStd = (np.concatenate((np.linspace(0.1,0.12,4), 0.12*np.ones(4), np.linspace(0.12,0.075,15), np.linspace(0.074,0.007,16), np.zeros(retired_T+1))))**0.5
TranShkStd = np.ndarray.tolist(TranShkStd)
PermShkStd = np.concatenate((((0.00011342*(np.linspace(24,64.75,working_T-1)-47)**2 + 0.01))**0.5,np.zeros(retired_T+1)))
PermShkStd[31:39] = PermShkStd[30] # Don't extrapolate permanent shock stdev
PermShkStd = np.ndarray.tolist(PermShkStd)
TranShkStdAllHealth = []
PermShkStdAllHealth = []
for t in range(AgeCount):
    TranShkStdAllHealth.append(5*[TranShkStd[t]])
    PermShkStdAllHealth.append(5*[PermShkStd[t]])

# Make an age-varying list of ESImrkvArray
ESImrkvArray_list = []
for t in range(working_T-1):
    ESImrkvArray_list.append(copy(ESImrkvArray))
ESImrkvArray_list.append(np.array([[1.],[1.],[1.]]))
for t in range(retired_T):
    ESImrkvArray_list.append(np.array([[1.]]))
    
# Make education-specific health transitions, estimated directly from the MEPS
f1 = lambda x :  .1780641*x - .0094783*x**2 + .0001773*x**3 - 1.12e-06*x**4
f2 = lambda x : -.0319569*x + .0006053*x**2 - 8.86e-06*x**3 + 5.25e-08*x**4
f3 = lambda x : -.0518241*x + .0014839*x**2 - .0000209*x**3 + 9.98e-08*x**4
f4 = lambda x : -.0300812*x + .0008112*x**2 - .0000106*x**3 + 3.50e-08*x**4
f5 = lambda x : -.0293762*x + .0007694*x**2 - 7.50e-06*x**3 + 2.58e-09*x**4
cuts1 = np.array([.2567907, 1.192993,  2.062196, 2.54124])
cuts2 = np.array([-2.247301,-.842383,.3368066,1.191534])
cuts3 = np.array([-2.840044,-1.851338,-.3702818,.7215166])
cuts4 = np.array([-2.921203,-2.215344,-.9558878,.4987979])
cuts5 = np.array([-3.030165,-2.535108,-1.487266,-.4969349])
educ_bonus = np.array([[.080082,0.,.1814115],
                       [-.0648348,0.,.0607891],
                       [-.0999835,0.,.169032],
                       [-.1128418,0.,.199002],
                       [-.2621089,0.,.1883597]])

# Fill in the Markov array at each age (probably could have written this more cleverly but meh)
MrkvArrayByEduc = np.zeros([67,5,5,3]) + np.nan
Age = np.arange(0,67,dtype=float) # This is age minus 18
for j in range(3):
    f1a = lambda x : f1(x) + educ_bonus[0,j]
    fitted = f1a(Age)
    MrkvArrayByEduc[:,0,0,j] = norm.cdf(cuts1[0] - fitted) - norm.cdf(-np.inf  - fitted)
    MrkvArrayByEduc[:,0,1,j] = norm.cdf(cuts1[1] - fitted) - norm.cdf(cuts1[0] - fitted)
    MrkvArrayByEduc[:,0,2,j] = norm.cdf(cuts1[2] - fitted) - norm.cdf(cuts1[1] - fitted)
    MrkvArrayByEduc[:,0,3,j] = norm.cdf(cuts1[3] - fitted) - norm.cdf(cuts1[2] - fitted)
    MrkvArrayByEduc[:,0,4,j] = norm.cdf(np.inf   - fitted) - norm.cdf(cuts1[3] - fitted)
    f2a = lambda x : f2(x) + educ_bonus[1,j]
    fitted = f2a(Age)
    MrkvArrayByEduc[:,1,0,j] = norm.cdf(cuts2[0] - fitted) - norm.cdf(-np.inf  - fitted)
    MrkvArrayByEduc[:,1,1,j] = norm.cdf(cuts2[1] - fitted) - norm.cdf(cuts2[0] - fitted)
    MrkvArrayByEduc[:,1,2,j] = norm.cdf(cuts2[2] - fitted) - norm.cdf(cuts2[1] - fitted)
    MrkvArrayByEduc[:,1,3,j] = norm.cdf(cuts2[3] - fitted) - norm.cdf(cuts2[2] - fitted)
    MrkvArrayByEduc[:,1,4,j] = norm.cdf(np.inf   - fitted) - norm.cdf(cuts2[3] - fitted)
    f3a = lambda x : f3(x) + educ_bonus[2,j]
    fitted = f3a(Age)
    MrkvArrayByEduc[:,2,0,j] = norm.cdf(cuts3[0] - fitted) - norm.cdf(-np.inf  - fitted)
    MrkvArrayByEduc[:,2,1,j] = norm.cdf(cuts3[1] - fitted) - norm.cdf(cuts3[0] - fitted)
    MrkvArrayByEduc[:,2,2,j] = norm.cdf(cuts3[2] - fitted) - norm.cdf(cuts3[1] - fitted)
    MrkvArrayByEduc[:,2,3,j] = norm.cdf(cuts3[3] - fitted) - norm.cdf(cuts3[2] - fitted)
    MrkvArrayByEduc[:,2,4,j] = norm.cdf(np.inf   - fitted) - norm.cdf(cuts3[3] - fitted)
    f4a = lambda x : f4(x) + educ_bonus[3,j]
    fitted = f4a(Age)
    MrkvArrayByEduc[:,3,0,j] = norm.cdf(cuts4[0] - fitted) - norm.cdf(-np.inf  - fitted)
    MrkvArrayByEduc[:,3,1,j] = norm.cdf(cuts4[1] - fitted) - norm.cdf(cuts4[0] - fitted)
    MrkvArrayByEduc[:,3,2,j] = norm.cdf(cuts4[2] - fitted) - norm.cdf(cuts4[1] - fitted)
    MrkvArrayByEduc[:,3,3,j] = norm.cdf(cuts4[3] - fitted) - norm.cdf(cuts4[2] - fitted)
    MrkvArrayByEduc[:,3,4,j] = norm.cdf(np.inf   - fitted) - norm.cdf(cuts4[3] - fitted)
    f5a = lambda x : f5(x) + educ_bonus[4,j]
    fitted = f5a(Age)
    MrkvArrayByEduc[:,4,0,j] = norm.cdf(cuts5[0] - fitted) - norm.cdf(-np.inf  - fitted)
    MrkvArrayByEduc[:,4,1,j] = norm.cdf(cuts5[1] - fitted) - norm.cdf(cuts5[0] - fitted)
    MrkvArrayByEduc[:,4,2,j] = norm.cdf(cuts5[2] - fitted) - norm.cdf(cuts5[1] - fitted)
    MrkvArrayByEduc[:,4,3,j] = norm.cdf(cuts5[3] - fitted) - norm.cdf(cuts5[2] - fitted)
    MrkvArrayByEduc[:,4,4,j] = norm.cdf(np.inf   - fitted) - norm.cdf(cuts5[3] - fitted)
    
# Make the array of health transitions after age 85
MrkvArrayOld = np.array([[0.450,0.367,0.183,0.000,0.000],
                        [0.140,0.445,0.290,0.115,0.010],
                        [0.060,0.182,0.497,0.208,0.053],
                        [0.032,0.090,0.389,0.342,0.147],
                        [0.000,0.061,0.231,0.285,0.423]])
    
# Reformat the Markov array into lifecycle lists by education, from age 18 to 120
MrkvArray_d = []
MrkvArray_h = []
MrkvArray_c = []
for t in range(67): # Until age 85
    MrkvArray_d.append(MrkvArrayByEduc[t,:,:,0])
    MrkvArray_h.append(MrkvArrayByEduc[t,:,:,1])
    MrkvArray_c.append(MrkvArrayByEduc[t,:,:,2])
for t in range(35): # Until age ~120
    MrkvArray_d.append(MrkvArrayOld)
    MrkvArray_h.append(MrkvArrayOld)
    MrkvArray_c.append(MrkvArrayOld)
MrkvArray_d = MrkvArray_d[7:] # Begin at age 25, dropping first 7 years
MrkvArray_h = MrkvArray_h[7:]
MrkvArray_c = MrkvArray_c[7:]

# Solve for survival probabilities at each health state for ages 24-60 by quasi-simulation
HealthDstnNow = np.array(HealthPrbsInit)
LivPrbYoung = np.zeros((5,60)) + np.nan
omega_vec = np.zeros(60)
HealthDstnHist = np.zeros((5,95))
for t in range(95):
    P = HealthDstnNow # For convenient typing
    DiePrbFunc = lambda q : P[0]*norm.cdf(q + HealthMortCum[3]) + P[1]*norm.cdf(q + HealthMortCum[2]) + P[2]*norm.cdf(q + HealthMortCum[1]) + P[3]*norm.cdf(q + HealthMortCum[0]) + P[4]*norm.cdf(q)
    LivPrbOut = lambda q : (1. - np.array([norm.cdf(q + HealthMortCum[3]),norm.cdf(q + HealthMortCum[2]),norm.cdf(q + HealthMortCum[1]),norm.cdf(q + HealthMortCum[0]),norm.cdf(q)]))**0.5
    ObjFunc = lambda q : DiePrbBiannual[t] - DiePrbFunc(q)
    if t < 60:
        omega_t = newton(ObjFunc,-3.)
        omega_vec[t] = omega_t
        LivPrbYoung[:,t] = LivPrbOut(omega_t)
        HealthDstnTemp = HealthDstnNow*LivPrbYoung[:,t] # Kill agents by health type
        HealthDstnNow = HealthDstnTemp/np.sum(HealthDstnTemp) # Renormalize to a stochastic vector
        HealthDstnNow = np.dot(HealthDstnNow,MrkvArrayByEduc[t+6,:,:,1]) # Apply health transitions to survivors
    else:
        HealthDstnTemp = HealthDstnNow*LivPrbOld[:,t-26] # Kill agents by health type
        HealthDstnNow = HealthDstnTemp/np.sum(HealthDstnTemp) # Renormalize to a stochastic vector
        HealthDstnNow = np.dot(HealthDstnNow,MrkvArrayOld[:,:]) # Apply health transitions to survivors
    HealthDstnHist[:,t] = HealthDstnNow
    
# Reformat LivPrb into a lifeycle list, from age 25 to 120
LivPrb = []
for t in range(working_T):
    LivPrb.append(LivPrbYoung[:,t+1])
for t in range(retired_T):
    LivPrb.append(LivPrbOld[:,t+15])
LivPrb[-1] = np.array([0.,0.,0.,0.,0.])

# Semi-arbitrary initial income levels (grab from data later)
pLvlInitMean_d = np.log(2.0)
pLvlInitMean_h = np.log(3.0)
pLvlInitMean_c = np.log(4.2)

# Permanent income growth rates from Cagetti (2003)
PermGroFac_d_base = [5.2522391e-002,  5.0039782e-002,  4.7586132e-002,  4.5162424e-002,  4.2769638e-002,  4.0408757e-002,  3.8080763e-002,  3.5786635e-002,  3.3527358e-002,  3.1303911e-002,  2.9117277e-002,  2.6968437e-002,  2.4858374e-002, 2.2788068e-002,  2.0758501e-002,  1.8770655e-002,  1.6825511e-002,  1.4924052e-002,  1.3067258e-002,  1.1256112e-002, 9.4915947e-003,  7.7746883e-003,  6.1063742e-003,  4.4876340e-003,  2.9194495e-003,  1.4028022e-003, -6.1326258e-005, -1.4719542e-003, -2.8280999e-003, -4.1287819e-003, -5.3730185e-003, -6.5598280e-003, -7.6882288e-003, -8.7572392e-003, -9.7658777e-003, -1.0713163e-002, -1.1598112e-002, -1.2419745e-002, -1.3177079e-002, -1.3869133e-002, -4.3985368e-001, -8.5623256e-003, -8.5623256e-003, -8.5623256e-003, -8.5623256e-003, -8.5623256e-003, -8.5623256e-003, -8.5623256e-003, -8.5623256e-003, -8.5623256e-003, -8.5623256e-003, -8.5623256e-003, -8.5623256e-003, -8.5623256e-003, -8.5623256e-003, -8.5623256e-003, -8.5623256e-003, -8.5623256e-003, -8.5623256e-003, -8.5623256e-003, -8.5623256e-003, -8.5623256e-003, -8.5623256e-003, -8.5623256e-003, -8.5623256e-003]
PermGroFac_h_base = [4.1102173e-002,  4.1194381e-002,  4.1117402e-002,  4.0878307e-002,  4.0484168e-002,  3.9942056e-002,  3.9259042e-002,  3.8442198e-002,  3.7498596e-002,  3.6435308e-002,  3.5259403e-002,  3.3977955e-002,  3.2598035e-002,  3.1126713e-002,  2.9571062e-002,  2.7938153e-002,  2.6235058e-002,  2.4468848e-002,  2.2646594e-002,  2.0775369e-002,  1.8862243e-002,  1.6914288e-002,  1.4938576e-002,  1.2942178e-002,  1.0932165e-002,  8.9156095e-003,  6.8995825e-003,  4.8911556e-003,  2.8974003e-003,  9.2538802e-004, -1.0178097e-003, -2.9251214e-003, -4.7894755e-003, -6.6038005e-003, -8.3610250e-003, -1.0054077e-002, -1.1675886e-002, -1.3219380e-002, -1.4677487e-002, -1.6043137e-002, -5.5864350e-001, -1.0820465e-002, -1.0820465e-002, -1.0820465e-002, -1.0820465e-002, -1.0820465e-002, -1.0820465e-002, -1.0820465e-002, -1.0820465e-002, -1.0820465e-002, -1.0820465e-002, -1.0820465e-002, -1.0820465e-002, -1.0820465e-002, -1.0820465e-002, -1.0820465e-002, -1.0820465e-002, -1.0820465e-002, -1.0820465e-002, -1.0820465e-002, -1.0820465e-002, -1.0820465e-002, -1.0820465e-002, -1.0820465e-002, -1.0820465e-002]
PermGroFac_c_base = [3.9375106e-002,  3.9030288e-002,  3.8601230e-002,  3.8091011e-002,  3.7502710e-002,  3.6839406e-002,  3.6104179e-002,  3.5300107e-002,  3.4430270e-002,  3.3497746e-002,  3.2505614e-002,  3.1456953e-002,  3.0354843e-002,  2.9202363e-002,  2.8002591e-002,  2.6758606e-002,  2.5473489e-002,  2.4150316e-002,  2.2792168e-002,  2.1402124e-002,  1.9983263e-002,  1.8538663e-002,  1.7071404e-002,  1.5584565e-002,  1.4081224e-002,  1.2564462e-002,  1.1037356e-002,  9.5029859e-003,  7.9644308e-003,  6.4247695e-003,  4.8870812e-003,  3.3544449e-003,  1.8299396e-003,  3.1664424e-004, -1.1823620e-003, -2.6640003e-003, -4.1251914e-003, -5.5628564e-003, -6.9739162e-003, -8.3552918e-003, -6.8938447e-001, -6.1023256e-004, -6.1023256e-004, -6.1023256e-004, -6.1023256e-004, -6.1023256e-004, -6.1023256e-004, -6.1023256e-004, -6.1023256e-004, -6.1023256e-004, -6.1023256e-004, -6.1023256e-004, -6.1023256e-004, -6.1023256e-004, -6.1023256e-004, -6.1023256e-004, -6.1023256e-004, -6.1023256e-004, -6.1023256e-004, -6.1023256e-004, -6.1023256e-004, -6.1023256e-004, -6.1023256e-004, -6.1023256e-004, -6.1023256e-004]
PermGroFac_d = PermGroFac_d_base[1:] + 31*[PermGroFac_d_base[-1]]
PermGroFac_h = PermGroFac_h_base[1:] + 31*[PermGroFac_h_base[-1]]
PermGroFac_c = PermGroFac_c_base[1:] + 31*[PermGroFac_c_base[-1]]
PermGroFac_dx = []
PermGroFac_hx = []
PermGroFac_cx = []
for t in range(working_T + retired_T):
    if t < working_T:
        PermGroFac_dx.append(5*[PermGroFac_d[t]+1.0])
        PermGroFac_hx.append(5*[PermGroFac_h[t]+1.0])
        PermGroFac_cx.append(5*[PermGroFac_c[t]+1.0])
    else:
        PermGroFac_dx.append(5*[1.0])
        PermGroFac_hx.append(5*[1.0])
        PermGroFac_cx.append(5*[1.0])

# Make retirement functions for each education level
SSbenefitFunc = LinearInterp([0.,1.062,6.4032,7.4032],[0.,0.9558,2.6650,2.8150])
LogAIMEfunc = lambda x : 0.5306 + 0.6932*x + 0.0368*x**2 - 0.0239*x**3
LogAIMEfuncDer = lambda x : 0.6932 + 0.0736*x - 0.0717*x**2
AIMEfunc = lambda x : np.exp(np.log(x))

class RetirementFunc(object):
    '''
    Function for representing pLvlNextFunc at retirement.
    '''
    low_point = 0.1  # Dollar value where we switch to lower extrap
    high_point = 20. # Dollar value where we switch to upper extrap
    
    def __init__(self, EducAdj, AIMEmax):
        log_high_point = np.log(self.high_point)
        log_low_point = np.log(self.low_point)
        self.AIMEmax = AIMEmax
        self.LogAIMEfunc = lambda x : LogAIMEfunc(x) + EducAdj
        self.low_slope = np.exp(self.LogAIMEfunc(log_low_point))/self.low_point
        self.high_slope = -LogAIMEfuncDer(log_high_point)
        if AIMEmax is not None:
            self.high_gap = np.log(AIMEmax) - self.LogAIMEfunc(log_high_point)
        else:
            self.high_intercept = self.LogAIMEfunc(log_high_point)
        
        
    def __call__(self, pLvlNow):
        if type(pLvlNow) is float:
            pLvlNow = np.array([pLvlNow])
        AIME = np.exp(self.LogAIMEfunc(np.log(pLvlNow)))
        low = pLvlNow < self.low_point
        AIME[low] = pLvlNow[low]*self.low_slope
        high = pLvlNow > self.high_point
        Diff = np.log(pLvlNow[high]) - np.log(self.high_point)
        if self.AIMEmax is not None:
            LogAIMEhigh = np.log(self.AIMEmax) - self.high_gap*np.exp(self.high_slope/self.high_gap*Diff)
        else:
            LogAIMEhigh = -self.high_slope*Diff + self.high_intercept
        AIME[high] = np.exp(LogAIMEhigh)
        pLvlNext = np.maximum(SSbenefitFunc(AIME),0.1)
        return pLvlNext

AIMEmax = None # Regular: 10.68
RetirementFunc_d = RetirementFunc(-0.0690, AIMEmax)
RetirementFunc_h = RetirementFunc(0.0000, AIMEmax)
RetirementFunc_c = RetirementFunc(0.0316, AIMEmax)
    
# Construct the probability of getting zero medical need shock by age-health
ZeroMedExFunc = lambda a : -0.2082171 - 0.0248579*a
ZeroMedShkPrb = np.zeros((95,5))
Age = np.arange(95)
ZeroMedShkPrb[:,4] = ZeroMedExFunc(Age)
ZeroMedShkPrb[:,3] = ZeroMedShkPrb[:,4] - 0.2081694 
ZeroMedShkPrb[:,2] = ZeroMedShkPrb[:,4] - 0.2503833 
ZeroMedShkPrb[:,1] = ZeroMedShkPrb[:,4] - 0.3943548 
ZeroMedShkPrb[:,0] = ZeroMedShkPrb[:,4] - 0.7203905 
ZeroMedShkPrb = norm.cdf(ZeroMedShkPrb)
ZeroMedShkPrb_list = []
for j in range(ZeroMedShkPrb.shape[0]):
    ZeroMedShkPrb_list.append(ZeroMedShkPrb[j,:])
    
# Make a basic dictionary with parameters that never change
BasicDictionary = { 'Rfree': Rfree,
                    'LivPrb': LivPrb,
                    'aXtraMin': aXtraMin,
                    'aXtraMax': aXtraMax,
                    'aXtraExtra': aXtraExtra,
                    'aXtraNestFac': aXtraNestFac,
                    'aXtraCount': aXtraCount,
                    'PermShkCount': PermShkCount,
                    'TranShkCount': TranShkCount,
                    'PermShkStd': PermShkStdAllHealth,
                    'TranShkStd': TranShkStdAllHealth,
                    'UnempPrb': UnempPrb,
                    'UnempPrbRet': UnempPrbRet,
                    'IncUnemp': IncUnemp,
                    'IncUnempRet': IncUnempRet,
                    'T_retire': working_T,
                    'BoroCnstArt': BoroCnstArt,
                    'CubicBool': CubicBool,
                    'DecurveBool': DecurveBool,
                    'pLvlPctiles': pLvlPctiles,
                    'pLvlInitStd': pLvlInitStd,
                    'pLvlInitMean': pLvlInitMean,
                    'PermIncCorr': PermIncCorr,
                    'MedShkCount': MedShkCount,
                    'DevMin' : DevMin,
                    'DevMax' : DevMax,
                    'ZeroMedShkPrb': ZeroMedShkPrb_list,
                    'MedPrice': T_cycle*[MedPrice],
                    'ESImrkvArray': ESImrkvArray_list,
                    'MrkvPrbsInit': HealthPrbsInit,
                    'T_cycle': T_cycle,
                    'T_sim': T_sim,
                    'AgentCount': AgentCount,
                    'do_sim': True,
                    'del_soln': True,
                    'verbosity': 0
                    }

# Make education-specific dictionaries
DropoutDictionary = copy(BasicDictionary)
DropoutDictionary['PermGroFac'] = PermGroFac_dx
DropoutDictionary['MrkvPrbsInit'] = MrkvPrbsInit_d
DropoutDictionary['HealthMrkvArray'] = MrkvArray_d
DropoutDictionary['pLvlInitMean'] = pLvlInitMean_d
DropoutDictionary['pLvlNextFuncRet'] = RetirementFunc_d
HighschoolDictionary = copy(BasicDictionary)
HighschoolDictionary['PermGroFac'] = PermGroFac_hx
HighschoolDictionary['MrkvPrbsInit'] = MrkvPrbsInit_h
HighschoolDictionary['HealthMrkvArray'] = MrkvArray_h
HighschoolDictionary['pLvlInitMean'] = pLvlInitMean_h
HighschoolDictionary['pLvlNextFuncRet'] = RetirementFunc_h
CollegeDictionary = copy(BasicDictionary)
CollegeDictionary['PermGroFac'] = PermGroFac_cx
CollegeDictionary['MrkvPrbsInit'] = MrkvPrbsInit_c
CollegeDictionary['HealthMrkvArray'] = MrkvArray_c
CollegeDictionary['pLvlInitMean'] = pLvlInitMean_c
CollegeDictionary['pLvlNextFuncRet'] = RetirementFunc_c

# Make a test parameter vector for estimation
test_param_vec = np.array([0.925,# DiscFac
                           2.8,  # CRRAcon
                           8.0,  # MedCurve 
                          -8.5,  # ChoiceShkMag in log
                           0.2,  # Cfloor
                         -1.51,  # SubsidyAvg
                          -3.0,  # SubsidyWidth scaler
                          10.0,  # BequestShift shifter for bequest motive
                           3.0,  # BequestScale scale of bequest motive
                         -3.45,  # MedShkMean constant coefficient
                        0.0045,  # MedShkMean linear age coefficient
                       0.00101,  # MedShkMean quadratic age coefficient
                    -0.0000018,  # MedShkMean cubic age coefficient
                   -0.00000012,  # MedShkMean quartic age coefficient
                          0.25,  # MedShkMean "very good" constant coefficient
                        0.0025,  # MedShkMean "very good" linear coefficient
                          0.30,  # MedShkMean "good" constant coefficient
                        0.0000,  # MedShkMean "good" linear coefficient
                          0.50,  # MedShkMean "fair" constant coefficient
                       -0.0011,  # MedShkMean "fair" linear coefficient
                          1.35,  # MedShkMean "poor" constant coefficient
                        -0.014,  # MedShkMean "poor" linear coefficient
                          0.40,  # MedShkStd constant coefficient
                        -0.001,  # MedShkStd linear age coefficient
                           0.0,  # MedShkStd quadratic age coefficient
                           0.0,  # MedShkStd cubic age coefficient
                           0.0,  # MedShkStd quartic age coefficient
                           0.0,  # MedShkStd "very good" constant coefficient
                           0.0,  # MedShkStd "very good" linear coefficient
                           0.04, # MedShkStd "good" constant coefficient
                           0.0,  # MedShkStd "good" linear coefficient
                           0.09, # MedShkStd "fair" constant coefficient
                           0.0,  # MedShkStd "fair" linear coefficient
                           0.02, # MedShkStd "poor" constant coefficient
                           0.0   # MedShkStd "poor" linear coefficient
                           ])

# This is very poor form, but I'm doing it anyway:
PremiumsLast = np.zeros(5)    