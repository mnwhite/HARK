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

# Set parameters for estimation
AgentCountTotal = 100000
StaticBool = True

# Calibrated / other parameters (grid sizes, etc)
Rfree = 5*[1.03]                    # Interest factor on assets
DiscFac = 0.96                      # Intertemporal discount factor
aXtraMin = 0.001                    # Minimum end-of-period "assets above minimum" value
aXtraMax = 80                       # Minimum end-of-period "assets above minimum" value               
aXtraExtra = [0.005,0.01]           # Some other value of "assets above minimum" to add to the grid, not used
aXtraNestFac = 3                    # Exponential nesting factor when constructing "assets above minimum" grid
aXtraCount = 32                    # Number of points in the grid of "assets above minimum"
PermShkCount = 7                    # Number of points in discrete approximation to permanent income shocks
TranShkCount = 7                    # Number of points in discrete approximation to transitory income shocks
UnempPrb = 0.05                     # Probability of unemployment while working
UnempPrbRet = 0.0005                # Probability of "unemployment" while retired
IncUnemp = 0.3                      # Unemployment benefits replacement rate
IncUnempRet = 0.0                   # "Unemployment" benefits when retired
BoroCnstArt = 0.0                   # Artificial borrowing constraint; imposed minimum level of end-of period assets
CubicBool = False                   # Use cubic spline interpolation when True, linear interpolation when False
PermIncCount = 12                   # Number of permanent income gridpoints in "body"
PermInc_tail_N = 3                  # Number of permanent income gridpoints in each "tail"
PermIncStdInit = 0.4                # Initial standard deviation of (log) permanent income (not used in example)
PermIncAvgInit = 1.0                # Initial average of permanent income (not used in example)
PermIncCorr = 1.0                   # Serial correlation coefficient for permanent income
MedShkCount = 5                     # Number of medical shock points in "body"
MedShkCountTail = [2,8]            # Number of medical shock points in "upper tail"
MedShkTailBound = [0.05,0.98]         # Boundaries of body (in CDF terms)
MedPrice = 1.0                      # Relative price of a unit of medical care
AgentCount = 10000                  # Number of agents of this type (only matters for simulation)
DeductibleList = [0.3,0.2,0.1,0.05,0.0] # List of deductibles for working-age insurance contracts
T_sim = 60                          # Number of periods to simulate (age 25 to 84)

# These are the results of ordered probits of h_t and age on h_t+1 using MEPS data
f1 = lambda x : .0579051*x -.0046128*x**2 + .0001069*x**3 - 7.85e-07*x**4
f2 = lambda x : -.0111249*x - .0010771*x**2 + .0000325*x**3 - 2.53e-07*x**4
f3 = lambda x : -.0068616*x - .0006085*x**2 + .0000169*x**3 - 1.35e-07*x**4
f4 = lambda x : -.0176393*x - .0002275*x**2 + .0000157*x**3 - 1.68e-07*x**4
f5 = lambda x : -.0002997*x -.0009072*x**2 + .0000311*x**3 - 2.96e-07*x**4
cuts1 = np.array([-.3176859, .5868993,1.416312,2.028111])
cuts2 = np.array([-2.028909,-.6592757,.4080535,1.13179])
cuts3 = np.array([-2.517836,-1.475572,-.0567525,.8971419])
cuts4 = np.array([-2.88919,-2.089985,-.9015471,.4327963])
cuts5 = np.array([-2.865663,-2.172939,-1.207311,-.3371267])

# Fill in the Markov array at each age (probably could have written this more cleverly but meh)
MrkvArrayYoung = np.zeros([67,5,5]) + np.nan
Age = np.arange(67,dtype=float)
fitted = f1(Age)
MrkvArrayYoung[:,0,0] = norm.cdf(cuts1[0] - fitted) - norm.cdf(-np.inf  - fitted)
MrkvArrayYoung[:,0,1] = norm.cdf(cuts1[1] - fitted) - norm.cdf(cuts1[0] - fitted)
MrkvArrayYoung[:,0,2] = norm.cdf(cuts1[2] - fitted) - norm.cdf(cuts1[1] - fitted)
MrkvArrayYoung[:,0,3] = norm.cdf(cuts1[3] - fitted) - norm.cdf(cuts1[2] - fitted)
MrkvArrayYoung[:,0,4] = norm.cdf(np.inf   - fitted) - norm.cdf(cuts1[3] - fitted)
fitted = f2(Age)
MrkvArrayYoung[:,1,0] = norm.cdf(cuts2[0] - fitted) - norm.cdf(-np.inf  - fitted)
MrkvArrayYoung[:,1,1] = norm.cdf(cuts2[1] - fitted) - norm.cdf(cuts2[0] - fitted)
MrkvArrayYoung[:,1,2] = norm.cdf(cuts2[2] - fitted) - norm.cdf(cuts2[1] - fitted)
MrkvArrayYoung[:,1,3] = norm.cdf(cuts2[3] - fitted) - norm.cdf(cuts2[2] - fitted)
MrkvArrayYoung[:,1,4] = norm.cdf(np.inf   - fitted) - norm.cdf(cuts2[3] - fitted)
fitted = f3(Age)
MrkvArrayYoung[:,2,0] = norm.cdf(cuts3[0] - fitted) - norm.cdf(-np.inf  - fitted)
MrkvArrayYoung[:,2,1] = norm.cdf(cuts3[1] - fitted) - norm.cdf(cuts3[0] - fitted)
MrkvArrayYoung[:,2,2] = norm.cdf(cuts3[2] - fitted) - norm.cdf(cuts3[1] - fitted)
MrkvArrayYoung[:,2,3] = norm.cdf(cuts3[3] - fitted) - norm.cdf(cuts3[2] - fitted)
MrkvArrayYoung[:,2,4] = norm.cdf(np.inf   - fitted) - norm.cdf(cuts3[3] - fitted)
fitted = f4(Age)
MrkvArrayYoung[:,3,0] = norm.cdf(cuts4[0] - fitted) - norm.cdf(-np.inf  - fitted)
MrkvArrayYoung[:,3,1] = norm.cdf(cuts4[1] - fitted) - norm.cdf(cuts4[0] - fitted)
MrkvArrayYoung[:,3,2] = norm.cdf(cuts4[2] - fitted) - norm.cdf(cuts4[1] - fitted)
MrkvArrayYoung[:,3,3] = norm.cdf(cuts4[3] - fitted) - norm.cdf(cuts4[2] - fitted)
MrkvArrayYoung[:,3,4] = norm.cdf(np.inf   - fitted) - norm.cdf(cuts4[3] - fitted)
fitted = f5(Age)
MrkvArrayYoung[:,4,0] = norm.cdf(cuts5[0] - fitted) - norm.cdf(-np.inf  - fitted)
MrkvArrayYoung[:,4,1] = norm.cdf(cuts5[1] - fitted) - norm.cdf(cuts5[0] - fitted)
MrkvArrayYoung[:,4,2] = norm.cdf(cuts5[2] - fitted) - norm.cdf(cuts5[1] - fitted)
MrkvArrayYoung[:,4,3] = norm.cdf(cuts5[3] - fitted) - norm.cdf(cuts5[2] - fitted)
MrkvArrayYoung[:,4,4] = norm.cdf(np.inf   - fitted) - norm.cdf(cuts5[3] - fitted)
# MrkvArrayYoung runs from age 18 to age 84

# Make the array of health transitions after age 85
MrkvArrayOld = np.array([[0.450,0.367,0.183,0.000,0.000],
                        [0.140,0.445,0.290,0.115,0.010],
                        [0.060,0.182,0.497,0.208,0.053],
                        [0.032,0.090,0.389,0.342,0.147],
                        [0.000,0.061,0.231,0.285,0.423]])
    
# Make survival probabilities by health state and age based on a probit on HRS data for ages 50-119
#AgeMortParams = [-2.881993,.0631395,-.0025132,.0000761,-6.27e-07] # MEN AND WOMEN
#HealthMortAdj = [.0384278,.231691,.3289953,.4941035]              # MEN AND WOMEN
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

# Specify the initial distribution of health at age 24 or 25, taken directly from MEPS data
HealthPrbsInit = [0.010,0.058,0.233,0.327,0.372]
HealthPrbsInit_d = [0.023,0.104,0.323,0.288,0.262]
HealthPrbsInit_h = [0.013,0.058,0.235,0.347,0.347]
HealthPrbsInit_c = [0.004,0.027,0.173,0.380,0.416]
EducWeight = [0.114,0.549,0.337]

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
        HealthDstnNow = np.dot(HealthDstnNow,MrkvArrayYoung[t+6,:,:]) # Apply health transitions to survivors
    else:
        HealthDstnTemp = HealthDstnNow*LivPrbOld[:,t-26] # Kill agents by health type
        HealthDstnNow = HealthDstnTemp/np.sum(HealthDstnTemp) # Renormalize to a stochastic vector
        HealthDstnNow = np.dot(HealthDstnNow,MrkvArrayOld[:,:]) # Apply health transitions to survivors
    HealthDstnHist[:,t] = HealthDstnNow
    
#plt.plot(np.cumsum(HealthDstnHist,axis=0).transpose())
#plt.show()

#plt.plot(HealthDstnHist.transpose())
#plt.show()

# Make the income shock standard deviations by age, from age 25-120
retired_T = 55
working_T = 40
AgeCount = retired_T + working_T
T_cycle = retired_T + working_T
TranShkStd = (np.concatenate((np.linspace(0.1,0.12,4), 0.12*np.ones(4), np.linspace(0.12,0.075,15), np.linspace(0.074,0.007,17), np.zeros(retired_T))))**0.5
TranShkStd = np.ndarray.tolist(TranShkStd)
PermShkStd = np.concatenate((((0.00011342*(np.linspace(24,64.75,working_T-1)-47)**2 + 0.01))**0.5,np.zeros(retired_T+1)))
PermShkStd = np.ndarray.tolist(PermShkStd)
TranShkStdAllHealth = []
PermShkStdAllHealth = []
for t in range(AgeCount):
    TranShkStdAllHealth.append(5*[TranShkStd[t]])
    PermShkStdAllHealth.append(5*[PermShkStd[t]])

#plt.plot(Age+50,MortProbitEX(Age))
#plt.plot(np.arange(24,84),omega_vec)
#plt.show()

# These are the results of ordered probits of h_t and age on h_t+1 using HRS data
#f1 = lambda x : -.0474234*x + .0041993*x**2 - .0001465*x**3 + 1.78e-06*x**4
#f2 = lambda x : .0156372*x - .0016657*x**2 + .0000403*x**3 - 2.43e-07*x**4
#f3 = lambda x : -.0360361*x + .0032427*x**2 - .0001282*x**3 + 1.56e-06*x**4
#f4 = lambda x : -.0267533*x + .002038*x**2 - .0000888*x**3 + 1.08e-06*x**4
#f5 = lambda x : -.0211791*x +.0020567*x**2 - .0000981*x**3 + 1.20e-06*x**4
#cuts1 = np.array([.0008348, .9714973, 1.636862, 2.213901])
#cuts2 = np.array([-1.04425,.3155104,1.303483,2.11849])
#cuts3 = np.array([-1.944034,-.9095382,.4746595,1.596067])
#cuts4 = np.array([-2.38825,-1.600396,-.5677765,.9822929])
#cuts5 = np.array([-2.462641,-1.833973,-1.10436,-.0848614])

# Reformat the Markov array into a lifecycle list, from age 18 to 120
MrkvArray = []
for t in range(67): # Until age 85
    MrkvArray.append(MrkvArrayYoung[t,:,:])
for t in range(35): # Until age ~120
    MrkvArray.append(MrkvArrayOld)
MrkvArray = MrkvArray[7:] # Begin at age 25, dropping first 7 years

# Reformat LivPrb into a lifeycle list, from age 25 to 120
LivPrb = []
#LivPrbYoung[:,:] = LivPrbYoung[4,:] # Shut off health-based mortality
#LivPrbOld[:,:] = LivPrbOld[4,:]
for t in range(40):
    LivPrb.append(LivPrbYoung[:,t+1])
for t in range(55):
    LivPrb.append(LivPrbOld[:,t+15])
    
# Make education-specific health transitions, estimated directly from the MEPS
f1 = lambda x :  .0573877*x - .0044485*x**2 + .0001006*x**3 - 7.23e-07*x**4
f2 = lambda x : -.0020471*x - .0016886*x**2 + .0000476*x**3 - 3.73e-07*x**4
f3 = lambda x : -.0280355*x + .0002937*x**2 + 1.80e-06*x**3 - 4.27e-08*x**4
f4 = lambda x : -.0394604*x + .0010916*x**2 - .0000139*x**3 + 5.17e-08*x**4
f5 = lambda x : -.032489*x  + .001016*x**2 - .0000131*x**3 + 4.30e-08*x**4
cuts1 = np.array([-.2374138, .6798592,1.49877,2.060003])
cuts2 = np.array([-2.004649,-.6456054,.4672089,1.24509])
cuts3 = np.array([-2.65231,-1.631041,-.1872363,.8339742])
cuts4 = np.array([-2.990194,-2.227373,-1.007045,.3960801])
cuts5 = np.array([-2.994232,-2.410726,-1.434513,-.4883896])
educ_bonus = np.array([[.0673213,0.,.1527079],
                       [-.0882254,0.,.042295],
                       [-.1054812,0.,.1704653],
                       [-.1477634,0.,.1994524],
                       [-.2401582,0.,.1918756]])

# Fill in the Markov array at each age (probably could have written this more cleverly but meh)
MrkvArrayByEduc = np.zeros([67,5,5,3]) + np.nan
Age = np.arange(67,dtype=float)
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

# Semi-arbitrary initial income levels (grab from data later)
PermIncAvgInit_d = 2.0
PermIncAvgInit_h = 3.0
PermIncAvgInit_c = 4.2

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
for t in range(95):
    PermGroFac_dx.append(5*[PermGroFac_d[t]+1.0])
    PermGroFac_hx.append(5*[PermGroFac_h[t]+1.0])
    PermGroFac_cx.append(5*[PermGroFac_c[t]+1.0])
    
    
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
                    'PermIncCount': PermIncCount,
                    'PermInc_tail_N': PermInc_tail_N,
                    'PermIncStdInit': PermIncStdInit,
                    'PermIncAvgInit': PermIncAvgInit,
                    'PermIncCorr': PermIncCorr,
                    'MedShkCount': MedShkCount,
                    'MedShkCountTail': MedShkCountTail,
                    'MedShkTailBound': MedShkTailBound,
                    'ZeroMedShkPrb': ZeroMedShkPrb,
                    'MedPrice': T_cycle*[MedPrice],
                    'MrkvArray': MrkvArray,
                    'MrkvPrbsInit': HealthPrbsInit,
                    'T_cycle': T_cycle,
                    'T_sim': T_sim,
                    'AgentCount': AgentCount,
                    }

# Make education-specific dictionaries
DropoutDictionary = copy(BasicDictionary)
DropoutDictionary['PermGroFac'] = PermGroFac_dx
DropoutDictionary['MrkvPrbsInit'] = HealthPrbsInit_d
DropoutDictionary['MrkvArray'] = MrkvArray_d
DropoutDictionary['PermIncAvgInit'] = PermIncAvgInit_d
HighschoolDictionary = copy(BasicDictionary)
HighschoolDictionary['PermGroFac'] = PermGroFac_hx
HighschoolDictionary['MrkvPrbsInit'] = HealthPrbsInit_h
HighschoolDictionary['MrkvArray'] = MrkvArray_h
HighschoolDictionary['PermIncAvgInit'] = PermIncAvgInit_h
CollegeDictionary = copy(BasicDictionary)
CollegeDictionary['PermGroFac'] = PermGroFac_cx
CollegeDictionary['MrkvPrbsInit'] = HealthPrbsInit_c
CollegeDictionary['MrkvArray'] = MrkvArray_c
CollegeDictionary['PermIncAvgInit'] = PermIncAvgInit_c

# Make a test parameter vector for estimation
test_param_vec = np.array([0.955, # DiscFac
                           2.6,  # CRRAcon
                           8.0,  # CRRAmed 
                          -7.0,  # ChoiceShkMag in log
                           2.0,  # SubsidyZeroRate scaler
                           0.0,  # SubsidyAvg
                           0.0,  # SubsidyWidth scaler
                          -5.5,  # MedShkMean constant coefficient
                         0.014,  # MedShkMean linear age coefficient
                        0.0005,  # MedShkMean quadratic age coefficient
                     -0.000000,  # MedShkMean cubic age coefficient
                   -0.00000000,  # MedShkMean quartic age coefficient
                          0.35,  # MedShkMean "very good" constant coefficient
                           0.0,  # MedShkMean "very good" linear coefficient
                           0.3,  # MedShkMean "good" constant coefficient
                           0.0,  # MedShkMean "good" linear coefficient
                          0.50,  # MedShkMean "fair" constant coefficient
                         -0.00,  # MedShkMean "fair" linear coefficient
                           1.3,  # MedShkMean "poor" constant coefficient
                         -0.01,  # MedShkMean "poor" linear coefficient
                           0.4,  # MedShkStd constant coefficient
                         0.001,  # MedShkStd linear age coefficient
                           0.0,  # MedShkStd quadratic age coefficient
                           0.0,  # MedShkStd cubic age coefficient
                           0.0,  # MedShkStd quartic age coefficient
                           0.0,  # MedShkStd "very good" constant coefficient
                           0.0,  # MedShkStd "very good" linear coefficient
                           0.0,  # MedShkStd "good" constant coefficient
                           0.0,  # MedShkStd "good" linear coefficient
                           0.0,  # MedShkStd "fair" constant coefficient
                           0.0,  # MedShkStd "fair" linear coefficient
                           0.0,  # MedShkStd "poor" constant coefficient
                           0.0   # MedShkStd "poor" linear coefficient
                           ])

# These are some parameters where things go crazy (only works with static model)
#test_param_vec = np.array([0.9, # DiscFac
#                           2.0,  # CRRAcon
#                           20.,  # CRRAmed
#                          -7.0,  # ChoiceShkMag in log
#                           2.0,  # SubsidyZeroRate scaler
#                           0.0,  # SubsidyAvg
#                           0.0,  # SubsidyWidth scaler
#                         -66.5,  # MedShkMean constant coefficient
#                          0.40,  # MedShkMean linear age coefficient
#                        0.0060,  # MedShkMean quadratic age coefficient
#                     -0.000004,  # MedShkMean cubic age coefficient
#                   -0.00000005,  # MedShkMean quartic age coefficient
#                           5.5,  # MedShkMean "very good" constant coefficient
#                           0.0,  # MedShkMean "very good" linear coefficient
#                           6.0,  # MedShkMean "good" constant coefficient
#                           0.0,  # MedShkMean "good" linear coefficient
#                           8.0,  # MedShkMean "fair" constant coefficient
#                         -0.00,  # MedShkMean "fair" linear coefficient
#                          25.0,  # MedShkMean "poor" constant coefficient
#                         -0.20,  # MedShkMean "poor" linear coefficient
#                          3.25,  # MedShkStd constant coefficient
#                         0.002,  # MedShkStd linear age coefficient
#                           0.0,  # MedShkStd quadratic age coefficient
#                           0.0,  # MedShkStd cubic age coefficient
#                           0.0,  # MedShkStd quartic age coefficient
#                           0.0,  # MedShkStd "very good" constant coefficient
#                           0.0,  # MedShkStd "very good" linear coefficient
#                           0.0,  # MedShkStd "good" constant coefficient
#                           0.0,  # MedShkStd "good" linear coefficient
#                           0.0,  # MedShkStd "fair" constant coefficient
#                           0.0,  # MedShkStd "fair" linear coefficient
#                           0.0,  # MedShkStd "poor" constant coefficient
#                           0.0   # MedShkStd "poor" linear coefficient
#                           ])