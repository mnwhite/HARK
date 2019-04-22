'''
This module contains various IMI subsidy policy structures.
'''
import numpy as np
from HARKinterpolation import ConstantFunction, LinearInterp
from HARKutilities import plotFuncs

T_working = 40
T_retired = 55
HealthCount = 5

# Make a null subsidy policy: no subsidies for anything
NoSubsidyFunc = ConstantFunction(0.0)
NoSubsidyFunc_t = HealthCount*[NoSubsidyFunc]
NullSubsidyFuncs = (T_working + T_retired)*[NoSubsidyFunc_t]

# Make a simple flat subsidy policy with constant value
FlatSubsidyFunc = ConstantFunction(0.1)
FlatSubsidyFunc_t = HealthCount*[FlatSubsidyFunc]
FlatSubsidyFuncs = T_working*[FlatSubsidyFunc_t] + T_retired*[NoSubsidyFunc_t]

class ACAsubsidyFunc():
    '''
    A class for representing ACA subsidy functions.
    
    Parameters
    ----------
    MaxPctOOP : float
        Highest percentage of income the individual will have to pay for IMI,
        assuming they are eligible for a subsidy.  This is 0.095 in the ACA.
    FPLpctCutoff : float
        Maximum income level (as percentage of the FPL) at which a household is
        eligible to receive any subsidy.  This is 4.0 in the ACA.
        
    Returns
    -------
    None
    '''
    def __init__(self,MaxPctOOP,FPLpctCutoff):
        FPL = 1.18 # 2014 value for single person
        x_base = np.array([0.000, 0.999, 1.00, 1.329, 1.33, 1.50, 2.00, 2.50])
        y_base = np.array([100., 100., 0.02, 0.02, 0.03, 0.04, 0.063, 0.080])
        kink = (MaxPctOOP - 0.08)*(100./3.) + 2.5
        x_extra = np.array([kink, FPLpctCutoff, FPLpctCutoff+0.001, FPLpctCutoff+1.0])
        y_extra = np.array([MaxPctOOP, MaxPctOOP, 100., 100.])
        x_vec = FPL*np.concatenate([x_base,x_extra])
        y_vec = np.concatenate([y_base,y_extra])
        z_vec = x_vec*y_vec
        z_vec[0:2] = 100.
        z_vec[-2:] = 100.
        MaxOOPpremiumFunc = LinearInterp(x_vec,z_vec)
        self.MaxOOPpremiumFunc = MaxOOPpremiumFunc
        
    def __call__(self,mLvl,yLvl,Premium):
        OOPpremCap = self.MaxOOPpremiumFunc(yLvl)
        Subsidy = np.maximum(Premium - OOPpremCap, 0.0)
        return Subsidy
    
    def derivativeX(self,mLvl,yLvl,Premium):
        return np.zeros_like(mLvl)
    
    def derivativeY(self,mLvl,yLvl,Premium):
        OOPpremCap = self.MaxOOPpremiumFunc(yLvl)
        Subsidy = np.maximum(Premium - OOPpremCap, 0.0)
        dPremCapdyLvl = self.MaxOOPpremiumFunc.derivative(yLvl)
        out = -dPremCapdyLvl
        out[Subsidy == 0.] = 0.
        return out
    
    def derivativeZ(self,mLvl,yLvl,Premium):
        OOPpremCap = self.MaxOOPpremiumFunc(yLvl)
        Subsidy = np.maximum(Premium - OOPpremCap, 0.0)
        out = np.ones_like(mLvl)
        out[Subsidy == 0.] = 0.
        return out
        

def makeACAstyleSubsidyPolicy(MaxPctOOP,FPLpctCutoff):
    '''
    Make an IMI subsidy policy in the style of the ACA, specifying a maximum
    out-of-pocket premium as a function of (permanent) income and providing a
    subsidy to make up the difference from the actual premium.
    
    Parameters
    ----------
    MaxPctOOP : float
        Highest percentage of income the individual will have to pay for IMI,
        assuming they are eligible for a subsidy.  This is 0.095 in the ACA.
    FPLpctCutoff : float
        Maximum income level (as percentage of the FPL) at which a household is
        eligible to receive any subsidy.  This is 4.0 in the ACA.
        
    Returns
    -------
    TheseACAsubsidyFuncs : [[ACAsubsidyFunc]]
        Nested list of IMI subsidy functions (by age and health status).
    '''
    ThisACAsubsidyFunc = ACAsubsidyFunc(MaxPctOOP,FPLpctCutoff)
    ThisACASubsidyFunc_t = HealthCount*[ThisACAsubsidyFunc]
    TheseACAsubsidyFuncs = T_working*[ThisACASubsidyFunc_t] + T_retired*[NoSubsidyFunc_t]
    return TheseACAsubsidyFuncs
    

# TODO: Make BCA-style subsidy policy
