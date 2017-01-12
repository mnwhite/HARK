'''
This module makes exogenous agent parameters for the DynInsSel project.
'''
import numpy as np
from scipy.stats import norm
from scipy.optimize import newton
import matplotlib.pyplot as plt
import os
import csv

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
    
plt.plot(np.cumsum(HealthDstnHist,axis=0).transpose())
plt.show()

plt.plot(HealthDstnHist.transpose())
plt.show()

# Make the income shock standard deviations by age, from age 25-120
retired_T = 55
working_T = 40
TranShkStd = (np.concatenate((np.linspace(0.1,0.12,4), 0.12*np.ones(4), np.linspace(0.12,0.075,15), np.linspace(0.074,0.007,17), np.zeros(retired_T))))**0.5
TranShkStd = np.ndarray.tolist(TranShkStd)
PermShkStd = np.concatenate((((0.00011342*(np.linspace(24,64.75,working_T-1)-47)**2 + 0.01))**0.5,np.zeros(retired_T+1)))
PermShkStd = np.ndarray.tolist(PermShkStd)

plt.plot(Age+50,MortProbitEX(Age))
plt.plot(np.arange(24,84),omega_vec)
plt.show()

# These are the results of ordered probits of h_t and age on h_t+1 using HRS data
f1 = lambda x : -.0474234*x + .0041993*x**2 - .0001465*x**3 + 1.78e-06*x**4
f2 = lambda x : .0156372*x - .0016657*x**2 + .0000403*x**3 - 2.43e-07*x**4
f3 = lambda x : -.0360361*x + .0032427*x**2 - .0001282*x**3 + 1.56e-06*x**4
f4 = lambda x : -.0267533*x + .002038*x**2 - .0000888*x**3 + 1.08e-06*x**4
f5 = lambda x : -.0211791*x +.0020567*x**2 - .0000981*x**3 + 1.20e-06*x**4
cuts1 = np.array([.0008348, .9714973, 1.636862, 2.213901])
cuts2 = np.array([-1.04425,.3155104,1.303483,2.11849])
cuts3 = np.array([-1.944034,-.9095382,.4746595,1.596067])
cuts4 = np.array([-2.38825,-1.600396,-.5677765,.9822929])
cuts5 = np.array([-2.462641,-1.833973,-1.10436,-.0848614])

# Fill in the Markov array at each age (probably could have written this more cleverly but meh)
#MrkvArrayOld = np.zeros([70,5,5]) + np.nan
#Age = np.arange(70,dtype=float)
#fitted = f1(Age)
#MrkvArrayOld[:,0,0] = norm.cdf(cuts1[0] - fitted) - norm.cdf(-np.inf  - fitted)
#MrkvArrayOld[:,0,1] = norm.cdf(cuts1[1] - fitted) - norm.cdf(cuts1[0] - fitted)
#MrkvArrayOld[:,0,2] = norm.cdf(cuts1[2] - fitted) - norm.cdf(cuts1[1] - fitted)
#MrkvArrayOld[:,0,3] = norm.cdf(cuts1[3] - fitted) - norm.cdf(cuts1[2] - fitted)
#MrkvArrayOld[:,0,4] = norm.cdf(np.inf   - fitted) - norm.cdf(cuts1[3] - fitted)
#fitted = f2(Age)
#MrkvArrayOld[:,1,0] = norm.cdf(cuts2[0] - fitted) - norm.cdf(-np.inf  - fitted)
#MrkvArrayOld[:,1,1] = norm.cdf(cuts2[1] - fitted) - norm.cdf(cuts2[0] - fitted)
#MrkvArrayOld[:,1,2] = norm.cdf(cuts2[2] - fitted) - norm.cdf(cuts2[1] - fitted)
#MrkvArrayOld[:,1,3] = norm.cdf(cuts2[3] - fitted) - norm.cdf(cuts2[2] - fitted)
#MrkvArrayOld[:,1,4] = norm.cdf(np.inf   - fitted) - norm.cdf(cuts2[3] - fitted)
#fitted = f3(Age)
#MrkvArrayOld[:,2,0] = norm.cdf(cuts3[0] - fitted) - norm.cdf(-np.inf  - fitted)
#MrkvArrayOld[:,2,1] = norm.cdf(cuts3[1] - fitted) - norm.cdf(cuts3[0] - fitted)
#MrkvArrayOld[:,2,2] = norm.cdf(cuts3[2] - fitted) - norm.cdf(cuts3[1] - fitted)
#MrkvArrayOld[:,2,3] = norm.cdf(cuts3[3] - fitted) - norm.cdf(cuts3[2] - fitted)
#MrkvArrayOld[:,2,4] = norm.cdf(np.inf   - fitted) - norm.cdf(cuts3[3] - fitted)
#fitted = f4(Age)
#MrkvArrayOld[:,3,0] = norm.cdf(cuts4[0] - fitted) - norm.cdf(-np.inf  - fitted)
#MrkvArrayOld[:,3,1] = norm.cdf(cuts4[1] - fitted) - norm.cdf(cuts4[0] - fitted)
#MrkvArrayOld[:,3,2] = norm.cdf(cuts4[2] - fitted) - norm.cdf(cuts4[1] - fitted)
#MrkvArrayOld[:,3,3] = norm.cdf(cuts4[3] - fitted) - norm.cdf(cuts4[2] - fitted)
#MrkvArrayOld[:,3,4] = norm.cdf(np.inf   - fitted) - norm.cdf(cuts4[3] - fitted)
#fitted = f5(Age)
#MrkvArrayOld[:,4,0] = norm.cdf(cuts5[0] - fitted) - norm.cdf(-np.inf  - fitted)
#MrkvArrayOld[:,4,1] = norm.cdf(cuts5[1] - fitted) - norm.cdf(cuts5[0] - fitted)
#MrkvArrayOld[:,4,2] = norm.cdf(cuts5[2] - fitted) - norm.cdf(cuts5[1] - fitted)
#MrkvArrayOld[:,4,3] = norm.cdf(cuts5[3] - fitted) - norm.cdf(cuts5[2] - fitted)
#MrkvArrayOld[:,4,4] = norm.cdf(np.inf   - fitted) - norm.cdf(cuts5[3] - fitted)
# MrkvArrayOld runs from age 50 to age 119

# Reformat the Markov array into a lifecycle list, from age 18 to 120
#MrkvArray = []
#for t in range(67): # Until age 85
#    MrkvArray.append(MrkvArrayYoung[t,:,:])
#for t in range(35): # Until age ~120
#    MrkvArray.append(MrkvArrayOld)