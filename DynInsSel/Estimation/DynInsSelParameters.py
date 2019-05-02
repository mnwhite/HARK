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
Rfree = 5*[1.02]                    # Interest factor on assets
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
PermIncCorr = 0.99                  # Serial correlation coefficient for permanent income
MedShkCount = 20                    # Number of medical shock points
DevMin = -3.0                       # Minimum standard deviations below MedShk mean
DevMax = 5.0                        # Maximum standard deviations above MedShk mean
MedPrice = 1.0                      # Relative price of a unit of medical care
AgentCount = 10000                  # Number of agents of this type (only matters for simulation)
DeductibleList = [0.06,0.05,0.04,0.03,0.02] # List of deductibles for working-age insurance contracts
T_sim = 60                          # Number of periods to simulate (age 25 to 84)
retired_T = 55
working_T = 40


class SpecialTaxFunction():
    '''
    A simple class for representing income taxes to fund health care.  It simply
    taxes all income above a given threshold at a particular rate.
    '''
    def __init__(self,threshold,rate):
        self.threshold = threshold
        self.rate = rate
        
    def __call__(self,yLvl):
        return np.maximum((yLvl-self.threshold)*self.rate, 0.0)


class PolynomialFunction():
    '''
    A simple class for representing polynomial functions.
    '''
    def __init__(self,coeffs):
        self.coeffs = coeffs
        self.N = len(coeffs)
        
    def __call__(self,x):
        try:
            shape_orig = x.shape
            z = x.flatten()
        except:
            shape_orig = None
            z = x
        
        out = np.zeros_like(z) + self.coeffs[-1]
        for n in range(self.N-2,-1,-1):
            out *= z
            out += self.coeffs[n]
        
        if shape_orig is not None:
            return np.reshape(out,shape_orig)
        else:
            return out
        
        
class ESImarkovFunction():
    '''
    A class for representing transition probabilities among ESI states as a function
    of (log) permanent income.
    '''
    def __init__(self,ESItoNoneFunc,NoneToESIfunc,ContrToFullFunc,FullToContrFunc,EmpContrShare):
        self.ESItoNoneFunc = ESItoNoneFunc
        self.NoneToESIfunc = NoneToESIfunc
        self.ContrToFullFunc = ContrToFullFunc
        self.FullToContrFunc = FullToContrFunc
        self.EmpContrShare = EmpContrShare
        
    def __call__(self,pLvl):
        '''
        Generate a 3D array of transition probabilities among ESI states by pLvl.
        '''
        try:
            N = pLvl.size
        except:
            N = 1
        pLog = np.log(pLvl)
        ESImrkvArray = np.zeros((N,3,3))
        
        # Get probabilities of transitioning out of ESI
        ESItoNone = norm.cdf(self.ESItoNoneFunc(pLog))
        StayESI = 1.0 - ESItoNone
        ESImrkvArray[:,1,0] = ESItoNone
        ESImrkvArray[:,2,0] = ESItoNone
        
        # Get probabilities of transitioning into ESI
        NoneToESI = norm.cdf(self.NoneToESIfunc(pLog))
        StayNone = 1.0 - NoneToESI
        ESImrkvArray[:,0,0] = StayNone
        ESImrkvArray[:,0,1] = NoneToESI*(1.0-self.EmpContrShare)
        ESImrkvArray[:,0,2] = NoneToESI*self.EmpContrShare
        
        # Get probabilities of switching from getting employer contribution
        ContrToFull = norm.cdf(self.ContrToFullFunc(pLog))
        StayContr = 1.0 - ContrToFull
        ESImrkvArray[:,2,1] = ContrToFull*StayESI
        ESImrkvArray[:,2,2] = StayContr*StayESI
        
        # get probabilities of switching from paying full premium for ESI
        FullToContr = norm.cdf(self.FullToContrFunc(pLog))
        StayFull = 1.0 - FullToContr
        ESImrkvArray[:,1,2] = FullToContr*StayESI
        ESImrkvArray[:,1,1] = StayFull*StayESI
        
        return ESImrkvArray
        
# Make a trivial array of transition probabilities among ESI states.
# Order: No ESI, ESI with no emp contribution, ESI with emp contribution
ESImrkvArray = np.array([[0.8,0.1,0.1],
                         [0.03,0.80,0.17],
                         [0.01,0.015,0.975]])
       
# Make multinomial probit transition functions among ESI states
ESItoNone_0  = -2.2417979
ESItoNone_a1 = .0693082
ESItoNone_a2 = -.0018354
ESItoNone_a3 = .0000154
ESItoNone_ageFunc = PolynomialFunction([ESItoNone_0, ESItoNone_a1, ESItoNone_a2, ESItoNone_a3])
ESItoNone_p1 = -.1293572
ESItoNone_p2 = -.0601291
ESItoNone_p3 = .0241346
ESItoNone_ap = -.0018197
ESItoNone_d  = .2843128
ESItoNone_h  = 0.0 # Omitted
ESItoNone_c  = -.1736794
NoneToESI_0  = -.4653491
NoneToESI_a1 = -.0693082
NoneToESI_a2 = .0018354
NoneToESI_a3 = -.0000154
NoneToESI_ageFunc = PolynomialFunction([NoneToESI_0, NoneToESI_a1, NoneToESI_a2, NoneToESI_a3])
NoneToESI_p1 = .1293572
NoneToESI_p2 = .0601291
NoneToESI_p3 =  -.0241346
NoneToESI_ap = .0018197
NoneToESI_d  = -.2843128
NoneToESI_h  = 0.0 # Omitted
NoneToESI_c  = .1736794
ContrToFull_0  = norm.ppf(0.04)
ContrToFull_a1 = 0.0
ContrToFull_a2 = 0.0
ContrToFull_a3 = 0.0
ContrToFull_ageFunc = PolynomialFunction([ContrToFull_0, ContrToFull_a1, ContrToFull_a2, ContrToFull_a3])
ContrToFull_p1 = 0.0
ContrToFull_p2 = 0.0
ContrToFull_p3 = 0.0
ContrToFull_ap = 0.0
ContrToFull_d  = 0.0
ContrToFull_h  = 0.0 # Omitted
ContrToFull_c  = 0.0
FullToContr_0  = norm.ppf(0.60)
FullToContr_a1 = 0.0
FullToContr_a2 = 0.0
FullToContr_a3 = 0.0
FullToContr_ageFunc = PolynomialFunction([FullToContr_0, FullToContr_a1, FullToContr_a2, FullToContr_a3])
FullToContr_p1 = 0.0
FullToContr_p2 = 0.0
FullToContr_p3 = 0.0
FullToContr_ap = 0.0
FullToContr_d  = 0.0
FullToContr_h  = 0.0 # Omitted
FullToContr_c  = 0.0
EmpContrShare_d = 0.5
EmpContrShare_h = 0.5
EmpContrShare_c = 0.5
ESItoNoneFuncs_d = []
NoneToESIfuncs_d = []
ContrToFullFuncs_d = []
FullToContrFuncs_d = []
ESItoNoneFuncs_h = []
NoneToESIfuncs_h = []
ContrToFullFuncs_h = []
FullToContrFuncs_h = []
ESItoNoneFuncs_c = []
NoneToESIfuncs_c = []
ContrToFullFuncs_c = []
FullToContrFuncs_c = []
for a in range(18,64):
    ESItoNone_base = ESItoNone_ageFunc(a)
    ESItoNoneFuncs_d.append(PolynomialFunction([ESItoNone_base + ESItoNone_d, ESItoNone_p1 + a*ESItoNone_ap, ESItoNone_p2, ESItoNone_p3]))
    ESItoNoneFuncs_h.append(PolynomialFunction([ESItoNone_base + ESItoNone_h, ESItoNone_p1 + a*ESItoNone_ap, ESItoNone_p2, ESItoNone_p3]))
    ESItoNoneFuncs_c.append(PolynomialFunction([ESItoNone_base + ESItoNone_c, ESItoNone_p1 + a*ESItoNone_ap, ESItoNone_p2, ESItoNone_p3]))
    NoneToESI_base = NoneToESI_ageFunc(a)
    NoneToESIfuncs_d.append(PolynomialFunction([NoneToESI_base + NoneToESI_d, NoneToESI_p1 + a*NoneToESI_ap, NoneToESI_p2, NoneToESI_p3]))
    NoneToESIfuncs_h.append(PolynomialFunction([NoneToESI_base + NoneToESI_h, NoneToESI_p1 + a*NoneToESI_ap, NoneToESI_p2, NoneToESI_p3]))
    NoneToESIfuncs_c.append(PolynomialFunction([NoneToESI_base + NoneToESI_c, NoneToESI_p1 + a*NoneToESI_ap, NoneToESI_p2, NoneToESI_p3]))
    ContrToFull_base = ContrToFull_ageFunc(a)
    ContrToFullFuncs_d.append(PolynomialFunction([ContrToFull_base + ContrToFull_d, ContrToFull_p1 + a*ContrToFull_ap, ContrToFull_p2, ContrToFull_p3]))
    ContrToFullFuncs_h.append(PolynomialFunction([ContrToFull_base + ContrToFull_h, ContrToFull_p1 + a*ContrToFull_ap, ContrToFull_p2, ContrToFull_p3]))
    ContrToFullFuncs_c.append(PolynomialFunction([ContrToFull_base + ContrToFull_c, ContrToFull_p1 + a*ContrToFull_ap, ContrToFull_p2, ContrToFull_p3]))
    FullToContr_base = FullToContr_ageFunc(a)
    FullToContrFuncs_d.append(PolynomialFunction([FullToContr_base + FullToContr_d, FullToContr_p1 + a*FullToContr_ap, FullToContr_p2, FullToContr_p3]))
    FullToContrFuncs_h.append(PolynomialFunction([FullToContr_base + FullToContr_h, FullToContr_p1 + a*FullToContr_ap, FullToContr_p2, FullToContr_p3]))
    FullToContrFuncs_c.append(PolynomialFunction([FullToContr_base + FullToContr_c, FullToContr_p1 + a*FullToContr_ap, FullToContr_p2, FullToContr_p3]))
ESImrkvFuncs_d = []
ESImrkvFuncs_h = []
ESImrkvFuncs_c = []    
for a in range(18,64):
    t = a-18
    ESImrkvFuncs_d.append(ESImarkovFunction(ESItoNoneFuncs_d[t],NoneToESIfuncs_d[t],ContrToFullFuncs_d[t],FullToContrFuncs_d[t],EmpContrShare_d))
    ESImrkvFuncs_h.append(ESImarkovFunction(ESItoNoneFuncs_h[t],NoneToESIfuncs_h[t],ContrToFullFuncs_h[t],FullToContrFuncs_h[t],EmpContrShare_h))
    ESImrkvFuncs_c.append(ESImarkovFunction(ESItoNoneFuncs_c[t],NoneToESIfuncs_c[t],ContrToFullFuncs_c[t],FullToContrFuncs_c[t],EmpContrShare_c))
    
# Make ESI transition functions for retirement
class RetirementESIfunction():
    def __init__(self):
        pass
    
    def __call__(self,pLvl):
        try:
            N = pLvl.size
        except:
            N = 1
        ESImrkvArray = np.ones((N,3,1))
        return ESImrkvArray
    
class RetiredESIfunction():
    def __init__(self):
        pass
    
    def __call__(self,pLvl):
        try:
            N = pLvl.size
        except:
            N = 1
        ESImrkvArray = np.ones((N,1,1))
        return ESImrkvArray
    
ESImrkvFuncs_retired = [RetirementESIfunction()] + retired_T*[RetiredESIfunction()]
ESImrkvFuncs_d = ESImrkvFuncs_d[7:46] + ESImrkvFuncs_retired
ESImrkvFuncs_h = ESImrkvFuncs_h[7:46] + ESImrkvFuncs_retired
ESImrkvFuncs_c = ESImrkvFuncs_c[7:46] + ESImrkvFuncs_retired
    
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
AgeCount = retired_T + working_T
T_cycle = retired_T + working_T
TranShkStd = (np.concatenate((np.linspace(0.1,0.12,4), 0.12*np.ones(4), np.linspace(0.12,0.075,15), np.linspace(0.074,0.047,16), np.zeros(retired_T+1))))**0.5
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
    
# Individual market premiums by age
IMIpremiums = np.array([[ 1.90720776,  2.04733647,  2.09829736,  2.17896955,  2.22399231,
         2.271415  ,  2.31307986,  2.38737238,  2.45785184,  2.49594084,
         2.61279515,  2.60154544,  2.65382462,  2.70512115,  2.80536537,
         2.88313992,  2.96702573,  3.03236017,  3.11056428,  3.1696871 ,
         3.23526486,  3.32615963,  3.39319149,  3.52281168,  3.64036025,
         3.68719623,  3.80061138,  3.89738029,  4.05530509,  4.11555093,
         4.22203937,  4.33874121,  4.43892108,  4.57171832,  4.74798875,
         4.80763363,  4.92758941,  5.00968932,  5.1773331 ,  5.32721074],
       [ 0.3350381 ,  0.34179464,  0.3520346 ,  0.36066132,  0.3676093 ,
         0.3807557 ,  0.38942585,  0.3989395 ,  0.40940091,  0.41464459,
         0.42477153,  0.44099948,  0.45216506,  0.46331762,  0.4760266 ,
         0.48985183,  0.51233712,  0.52757109,  0.54808677,  0.56782115,
         0.59195688,  0.61695224,  0.64489221,  0.67384281,  0.70496672,
         0.73445768,  0.77984304,  0.80981509,  0.85208137,  0.90039156,
         0.93316004,  0.98088127,  1.02657617,  1.07069898,  1.11196758,
         1.15387465,  1.19883363,  1.24260055,  1.26661215,  1.30059763]])

# Define basic parameters of the economy
HealthTaxFunc = SpecialTaxFunction(0.0,0.00) # Tax rate will be overwritten by installPremiumFuncs
HealthTaxRate_init = 0.046
LoadFacESI   = 1.20 # Loading factor for employer sponsored insurance
LoadFacIMI   = 1.80 # Loading factor for individual market insurance
CohortGroFac = 1.01 # Year-on-year growth rate of population; each cohort is this factor larger than previous
    
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
                    'TaxFunc': HealthTaxFunc,
                    'MedShkCount': MedShkCount,
                    'DevMin' : DevMin,
                    'DevMax' : DevMax,
                    'ZeroMedShkPrb': ZeroMedShkPrb_list,
                    'MedPrice': T_cycle*[MedPrice],
                    'ESImrkvArray': ESImrkvArray_list,
                    'MrkvPrbsInit': HealthPrbsInit,
                    'HealthTaxRate': HealthTaxRate_init,
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
DropoutDictionary['ESImrkvFunc'] = ESImrkvFuncs_d
DropoutDictionary['pLvlInitMean'] = pLvlInitMean_d
DropoutDictionary['pLvlNextFuncRet'] = RetirementFunc_d
HighschoolDictionary = copy(BasicDictionary)
HighschoolDictionary['PermGroFac'] = PermGroFac_hx
HighschoolDictionary['MrkvPrbsInit'] = MrkvPrbsInit_h
HighschoolDictionary['HealthMrkvArray'] = MrkvArray_h
HighschoolDictionary['ESImrkvFunc'] = ESImrkvFuncs_h
HighschoolDictionary['pLvlInitMean'] = pLvlInitMean_h
HighschoolDictionary['pLvlNextFuncRet'] = RetirementFunc_h
CollegeDictionary = copy(BasicDictionary)
CollegeDictionary['PermGroFac'] = PermGroFac_cx
CollegeDictionary['MrkvPrbsInit'] = MrkvPrbsInit_c
CollegeDictionary['HealthMrkvArray'] = MrkvArray_c
CollegeDictionary['ESImrkvFunc'] = ESImrkvFuncs_c
CollegeDictionary['pLvlInitMean'] = pLvlInitMean_c
CollegeDictionary['pLvlNextFuncRet'] = RetirementFunc_c

# Make a test parameter vector for estimation
test_param_vec = np.array([
          0.88,                #  0 DiscFac           : Intertemporal discount factor
          2.9,                 #  1 CRRA              : Coefficient of relative risk aversion for consumption
          8.0,                 #  2 MedCurve          : Ratio of CRRA for medical care to CRRA for consumption
          -7.5,                #  3 log(ChoiceShkMag) : Log stdev of taste shocks over insurance contracts
          0.15,                #  4 Cfloor            : Consumption floor ($10,000)
          -1.43,               #  5 log(EmpContr)     : Log of employer contribution to ESI ($10,000)
          -3.0,                #  6 UNUSED            : UNUSED
          10.0,                #  7 BequestShift      : Constant term in bequest motive
          3.0,                 #  8 BequestScale      : Scale of bequest motive
          -3.47,               #  9 MedShkMean_0      : Constant term for log mean medical need shock
          0.0045,              # 10 MedShkMean_a1     : Linear coefficient on age for log mean medical need shock
          0.00075,             # 11 MedShkMean_a2     : Quadratic coefficient on age for log mean medical need shock
          8e-06,               # 12 MedShkMean_a3     : Cubic coefficient on age for log mean medical need shock
          -2.1e-07,            # 13 MedShkMean_a4     : Quartic coefficient on age for log mean medical need shock
          0.25,                # 14 MedShkMean_VG0    : Very good health shifter for log mean medical need shock
          0.0025,              # 15 MedShkMean_VGa1   : Very good health linear age coefficient shifter for log mean med shock
          0.53,                # 16 MedShkMean_GD0    : Good health shifter for log mean medical need shock
          0.0025,              # 17 MedShkMean_GDa1   : Good health linear age coefficient shifter for log mean med shock
          0.98,                # 18 MedShkMean_FR0    : Fair health shifter for log mean medical need shock
          0.0014,              # 19 MedShkMean_FRa1   : Fair health linear age coefficient shifter for log mean med shock
          2.18,                # 20 MedShkMean_PR0    : Poor health shifter for log mean medical need shock
          -0.0106,             # 21 MedShkMean_PRa1   : Poor health linear age coefficient shifter for log mean med shock
          0.338,               # 22 MedShkStd_0       : Constant term for log stdev medical need shock
          0.0065,              # 23 MedShkStd_a1      : Linear coefficient on age for log stdev medical need shock
          -0.00014,            # 24 MedShkStd_a2      : Quadratic coefficient on age for log stdev medical need shock
          0.0,                 # 25 MedShkStd_a3      : Cubic coefficient on age for log stdev medical need shock
          0.0,                 # 26 MedShkStd_a4      : Quartic coefficient on age for log stdev medical need shock
          0.041,               # 27 MedShkStd_VG0     : Very good health shifter for log stdev medical need shock
          0.0,                 # 28 MedShkStd_VGa1    : Very good health linear age coefficient shifter for log stdev med shock
          0.08,                # 29 MedShkStd_GD0     : Good health shifter for log stdev medical need shock
          0.0,                 # 30 MedShkStd_GDa1    : Good health linear age coefficient shifter for log stdev med shock
          0.19,                # 31 MedShkStd_FR0     : Fair health shifter for log stdev medical need shock
         -0.0013,              # 32 MedShkStd_FRa1    : Fair health linear age coefficient shifter for log stdev med shock
          0.27,                # 33 MedShkStd_PR0     : Poor health shifter for log stdev medical need shock
         -0.0020,              # 34 MedShkStd_PRa1    : Poor health linear age coefficient shifter for log stdev med shock
])


# This is very poor form, but I'm doing it anyway:
PremiumsLast = np.zeros(5)    