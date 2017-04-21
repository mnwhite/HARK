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
    # Order of indices: age, health, contract, type
    
    TotalInsPay = np.sum(ExpInsPayX[0:40,:,:,:],axis=(0,1,3))
    TotalBuyers = np.sum(ExpBuyersX[0:40,:,:,:],axis=(0,1,3))
    AvgInsPay   = TotalInsPay/TotalBuyers
    DampingFac = 0.2
    NewPremiums = (1.0-DampingFac)*self.LoadFac*AvgInsPay + DampingFac*self.Premiums
    NewPremiums[0] = 0.0 # First contract is always free
    print(NewPremiums)
    print(TotalBuyers/np.sum(TotalBuyers))

    PremiumFuncBase = []
    for z in range(MaxContracts):
        PremiumFuncBase.append(ConstantFunction(NewPremiums[z]))
    ZeroPremiumFunc = ConstantFunction(0.0)
        
    PremiumFuncs = 40*[StateCount*[PremiumFuncBase]] + 20*[StateCount*[MaxContracts*[ZeroPremiumFunc]]]
    self.Premiums = NewPremiums    
    return PremiumFuncsContainer(PremiumFuncs)


def exclusionaryActuarialRule(self,ExpInsPay,ExpBuyers):
    '''
    Constructs a nested list of premium functions that have a constant value,
    based on the average medical needs of all contract buyers.  Agents in some
    discrete states are prevented from buying any non-null contract and are
    offered a price of (effectively) infinity.
    
    Parameters
    ----------
    None
    
    Returns
    -------
    None
    '''
    ExcludedStates = [True, False, False, False, False] # Make this a market attribute later
    InfPrice = 10000.0
    
    StateCount = ExpInsPay[0].shape[1]
    MaxContracts = ExpInsPay[0].shape[2]
    ExpInsPayX = np.stack(ExpInsPay,axis=3) # This is collected in reap_vars
    ExpBuyersX = np.stack(ExpBuyers,axis=3) # This is collected in reap_vars
    # Order of indices: age, health, contract, type
    
    TotalInsPay = np.sum(ExpInsPayX[0:40,:,:,:],axis=(0,1,3))
    TotalBuyers = np.sum(ExpBuyersX[0:40,:,:,:],axis=(0,1,3))
    AvgInsPay   = TotalInsPay/TotalBuyers
    DampingFac = 0.2
    NewPremiums = (1.0-DampingFac)*self.LoadFac*AvgInsPay + DampingFac*self.Premiums
    NewPremiums[0] = 0.0 # First contract is always free
    print(NewPremiums)
    print(TotalBuyers/np.sum(TotalBuyers))

    PremiumFuncBase = []
    for z in range(MaxContracts):
        PremiumFuncBase.append(ConstantFunction(NewPremiums[z]))
    ZeroPremiumFunc = ConstantFunction(0.0)
    ExcludedPremiumFuncs = [ZeroPremiumFunc] + (MaxContracts-1)*[ConstantFunction(InfPrice)]
    
    WorkingAgePremiumFuncs = []
    for j in range(StateCount):
        if ExcludedStates[j]:
            WorkingAgePremiumFuncs.append(ExcludedPremiumFuncs)
        else:
            WorkingAgePremiumFuncs.append(PremiumFuncBase)
        
    PremiumFuncs = 40*[WorkingAgePremiumFuncs] + 20*[StateCount*[MaxContracts*[ZeroPremiumFunc]]]
    self.Premiums = NewPremiums    
    return PremiumFuncsContainer(PremiumFuncs)

    
def healthRatedActuarialRule(self,ExpInsPay,ExpBuyers):
    '''
    Constructs a nested list of premium functions that have a constant value,
    based on the average medical needs of contract buyers in the same health state.
    
    Parameters
    ----------
    None
    
    Returns
    -------
    None
    '''
    HealthStateGroups = [[0],[1],[2],[3],[4]]
    #HealthStateGroups = [[0,1],[2,3,4]]
    #HealthStateGroups = [[0,1,2,3,4]]
    GroupCount = len(HealthStateGroups)
    
    StateCount = ExpInsPay[0].shape[1]
    MaxContracts = ExpInsPay[0].shape[2]
    ExpInsPayX = np.stack(ExpInsPay,axis=3) # This is collected in reap_vars
    ExpBuyersX = np.stack(ExpBuyers,axis=3) # This is collected in reap_vars
    # Order of indices: age, health, contract, type
    
    PremiumArray = np.zeros((MaxContracts,GroupCount)) + np.nan
    for g in range(GroupCount):
        these = HealthStateGroups[g]
        TotalInsPay = np.sum(ExpInsPayX[0:40,these,:,:],axis=(0,1,3))
        TotalBuyers = np.sum(ExpBuyersX[0:40,these,:,:],axis=(0,1,3))
        AvgInsPay   = TotalInsPay/TotalBuyers
        DampingFac = 0.2
        NewPremiums = (1.0-DampingFac)*self.LoadFac*AvgInsPay + DampingFac*self.Premiums
        NewPremiums[0] = 0.0 # First contract is always free
        PremiumArray[:,g] = NewPremiums
        print(NewPremiums)
        
    TotalInsPay = np.sum(ExpInsPayX[0:40,:,:,:],axis=(0,1,3))
    TotalBuyers = np.sum(ExpBuyersX[0:40,:,:,:],axis=(0,1,3))
    print(TotalBuyers/np.sum(TotalBuyers))

    WorkingAgePremiumFuncs = StateCount*[None]
    for g in range(GroupCount):
        TempList = []
        for z in range(MaxContracts):
            TempList.append(ConstantFunction(PremiumArray[z,g]))
        for j in HealthStateGroups[g]:
            WorkingAgePremiumFuncs[j] = TempList
    ZeroPremiumFunc = ConstantFunction(0.0)
        
    PremiumFuncs = 40*[WorkingAgePremiumFuncs] + 20*[StateCount*[MaxContracts*[ZeroPremiumFunc]]]
    self.Premiums = NewPremiums    
    return PremiumFuncsContainer(PremiumFuncs)