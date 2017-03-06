'''
This module contains various "actuarial rules" for calculating premium functions
from the medical costs of consumers.
'''
import numpy as np
from HARKcore import HARKobject
from HARKinterpolation import ConstantFunction

# This is a trivial "container" class
class PremiumFuncsContainer(HARKobject):
    distance_criteria = ['PremiumFuncs']
    
    def __init__(self,PremiumFuncs):
        self.PremiumFuncs = PremiumFuncs

def flatActuarialRule(self,ExpInsPay,ExpBuyers):
    '''
    Constructs a nested list of premium functions that have a constant value,
    based on the average medical needs of all contract buyers.
    
    Parameters
    ----------
    None
    
    Returns
    -------
    None
    '''
    StateCount = ExpInsPay[0].shape[1]
    MaxContracts = ExpInsPay[0].shape[2]
    ExpInsPayX = np.stack(ExpInsPay,axis=3) # This is collected in reap_vars
    ExpBuyersX = np.stack(ExpBuyers,axis=3) # This is collected in reap_vars
    
    TotalInsPay = np.sum(ExpInsPayX[0:40,:,:,:],axis=(0,1,3))
    TotalBuyers = np.sum(ExpBuyersX[0:40,:,:,:],axis=(0,1,3))
    AvgInsPay   = TotalInsPay/TotalBuyers
    NewPremiums = self.LoadFac*AvgInsPay
    print(NewPremiums)

    PremiumFuncBase = []
    for z in range(MaxContracts):
        PremiumFuncBase.append(ConstantFunction(NewPremiums[z]))
    ZeroPremiumFunc = ConstantFunction(0.0)
        
    PremiumFuncs = 40*[StateCount*[PremiumFuncBase]] + 20*[StateCount*[MaxContracts*[ZeroPremiumFunc]]]
    self.Premiums = NewPremiums    
    return PremiumFuncsContainer(PremiumFuncs)
    