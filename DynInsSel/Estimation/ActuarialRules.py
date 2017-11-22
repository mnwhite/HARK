'''
This module contains various "actuarial rules" for calculating premium functions
from the medical costs of consumers.
'''
import numpy as np
from HARKcore import HARKobject
from HARKinterpolation import ConstantFunction
from copy import copy, deepcopy
from scipy.optimize import brentq

# This is a trivial "container" class
class PremiumFuncsContainer(HARKobject):
    distance_criteria = ['PremiumFuncs']
    
    def __init__(self,PremiumFuncs):
        self.PremiumFuncs = PremiumFuncs

def flatActuarialRule(self,ExpInsPay,ExpBuyers):
    '''
    Constructs a nested list of premium functions that have a constant value,
    based on the average medical needs of all contract buyers.  This is the
    baseline for the ESI market.
    
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
    based on the average medical needs of contract buyers in the same health
    state group; the groups are flexible.
    
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
        try:
            NewPremiums = (1.0-DampingFac)*self.LoadFac*AvgInsPay + DampingFac*self.Premiums[:,HealthStateGroups[g][0]]
        except:
            NewPremiums = self.LoadFac*AvgInsPay
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
    self.Premiums = PremiumArray    
    return PremiumFuncsContainer(PremiumFuncs)
    

def ageHealthRatedActuarialRule(self,ExpInsPay,ExpBuyers):
    '''
    Constructs a nested list of premium functions that have a constant value,
    based on the average medical needs of contract buyers in the same health
    state group and age; the health groups are defined flexibly.  This is the
    "before ACA" baseline for the individual market.
    
    Parameters
    ----------
    None
    
    Returns
    -------
    None
    '''
    #HealthStateGroups = [[0],[1],[2],[3],[4]]
    HealthStateGroups = [[0,1],[2,3,4]]
    #HealthStateGroups = [[0,1,2,3,4]]
    GroupCount = len(HealthStateGroups)
    AgeCount = 40
    
    StateCount = ExpInsPay[0].shape[1]
    MaxContracts = ExpInsPay[0].shape[2]
    ExpInsPayX = np.stack(ExpInsPay,axis=3) # This is collected in reap_vars
    ExpBuyersX = np.stack(ExpBuyers,axis=3) # This is collected in reap_vars
    # Order of indices: age, health, contract, type
    
    PremiumArray = np.zeros((MaxContracts,GroupCount,AgeCount)) + np.nan
    temp = []
    for z in range(MaxContracts):
        temp.append(None)
    temp2 = []
    for j in range(StateCount):
        temp2.append(copy(temp))
    WorkingAgePremiumFuncs = []
    for a in range(AgeCount):
        WorkingAgePremiumFuncs.append(deepcopy(temp2))
    
    for g in range(GroupCount):
        these = HealthStateGroups[g]
        TotalInsPay = np.sum(ExpInsPayX[0:40,these,:,:],axis=(1,3)) # Don't sum across ages
        TotalBuyers = np.sum(ExpBuyersX[0:40,these,:,:],axis=(1,3)) # Don't sum across ages
        AvgInsPay   = TotalInsPay/TotalBuyers
        DampingFac = 0.2
        try:
            NewPremiums = (1.0-DampingFac)*self.LoadFac*AvgInsPay + DampingFac*self.Premiums[:,HealthStateGroups[g][0],:]
        except:
            NewPremiums = self.LoadFac*AvgInsPay
        NewPremiums[:,0] = 0.0 # First contract is always free
        PremiumArray[:,g,:] = NewPremiums.transpose()

        for z in range(MaxContracts):
            for a in range(AgeCount):
                ThisPremiumFunc = ConstantFunction(PremiumArray[z,g,a])
                for j in HealthStateGroups[g]:
                    WorkingAgePremiumFuncs[a][j][z] = ThisPremiumFunc
        
    TotalInsPay = np.sum(ExpInsPayX[0:40,:,:,:],axis=(0,1,3))
    TotalBuyers = np.sum(ExpBuyersX[0:40,:,:,:],axis=(0,1,3))
    print(TotalBuyers/np.sum(TotalBuyers))

    ZeroPremiumFunc = ConstantFunction(0.0)        
    PremiumFuncs = WorkingAgePremiumFuncs + 20*[StateCount*[MaxContracts*[ZeroPremiumFunc]]]
    self.Premiums = PremiumArray    
    return PremiumFuncsContainer(PremiumFuncs)
    

def ageRatedActuarialRule(self,ExpInsPay,ExpBuyers):
    '''
    Constructs a nested list of premium functions that have a constant value,
    based on the average medical needs of contract buyers of the same age.
    An age-rating function is specified, which maps from the range of ages to
    the [0,1] interval.
    
    Parameters
    ----------
    None
    
    Returns
    -------
    None
    '''
    AgeRatingFunc = lambda x : (np.exp(x/40.*3.0)-1.)/(np.exp(3.0)-1.)
    AgeBandLimit = 5.0
    AgeCount = 40
    
    StateCount = ExpInsPay[0].shape[1]
    MaxContracts = ExpInsPay[0].shape[2]
    ExpInsPayX = np.stack(ExpInsPay,axis=3) # This is collected in reap_vars
    ExpBuyersX = np.stack(ExpBuyers,axis=3) # This is collected in reap_vars
    # Order of indices: age, health, contract, type
    
    PremiumArray = np.zeros((MaxContracts,AgeCount))
    temp = []
    for z in range(MaxContracts):
        temp.append(None)
    temp2 = []
    for j in range(StateCount):
        temp2.append(copy(temp))
    WorkingAgePremiumFuncs = []
    for a in range(AgeCount):
        WorkingAgePremiumFuncs.append(deepcopy(temp2))
    
    TotalInsPay = np.sum(ExpInsPayX[0:40,:,:,:],axis=(0,1,3))
    TotalBuyers = np.sum(ExpBuyersX[0:40,:,:,:],axis=(0,1,3))
    TotalBuyersByAge = np.sum(ExpBuyersX[0:40,:,:,:],axis=(1,3)) # Don't sum across ages
    AvgInsPay = TotalInsPay/TotalBuyers
    print(TotalBuyers/np.sum(TotalBuyers))
    
    AgeRatingScale = 1.0 + AgeBandLimit*AgeRatingFunc(np.arange(AgeCount,dtype=float))
    for z in range(1,MaxContracts):
        def tempFunc(BasePremium):
            PremiumVec = BasePremium*AgeRatingScale
            TotalRevenue = np.sum(PremiumVec*TotalBuyersByAge[:,z])
            return TotalRevenue/TotalInsPay[z] - self.LoadFac
        
        NewPremium = brentq(tempFunc,0.0,AvgInsPay[z]*self.LoadFac)
        PremiumArray[z,:] = NewPremium*AgeRatingScale
        
    for z in range(MaxContracts):
        for a in range(AgeCount):
            ThisPremiumFunc = ConstantFunction(PremiumArray[z,a])
            for j in range(StateCount):
                WorkingAgePremiumFuncs[a][j][z] = ThisPremiumFunc

    ZeroPremiumFunc = ConstantFunction(0.0)        
    PremiumFuncs = WorkingAgePremiumFuncs + 20*[StateCount*[MaxContracts*[ZeroPremiumFunc]]]
    self.Premiums = PremiumArray    
    return PremiumFuncsContainer(PremiumFuncs)
    
