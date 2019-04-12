'''
This module contains various "actuarial rules" for calculating premium functions
from the medical costs of consumers.
'''
import numpy as np
from HARKcore import HARKobject, Market
from HARKinterpolation import ConstantFunction
from copy import copy, deepcopy
from scipy.optimize import brentq

# This is a trivial "container" class
class PremiumFuncsContainer(HARKobject):
    distance_criteria = ['PremiumFuncs']
    
    def __init__(self,PremiumFuncs):
        self.PremiumFuncs = PremiumFuncs
        
        
class InsuranceMarket(Market):
    '''
    A class for representing the "insurance economy" with many agent types.
    '''

    def __init__(self,ActuarialRule):
        Market.__init__(self,agents=[],sow_vars=['PremiumFuncs'],
                        reap_vars=['ExpInsPay','ExpBuyers'],
                        const_vars=[],
                        track_vars=['Premiums'],
                        dyn_vars=['PremiumFuncs'],
                        millRule=None,calcDynamics=None,act_T=10,tolerance=0.0001)
        self.IMIactuarialRule = ActuarialRule

    def millRule(self,ExpInsPay,ExpBuyers):
        IMIpremiums     = self.IMIactuarialRule(self,ExpInsPay,ExpBuyers)
        ESIpremiums     = self.ESIactuarialRule(ExpInsPay,ExpBuyers)
        return self.combineESIandIMIpremiums(IMIpremiums,ESIpremiums)
        
    def calcDynamics(self,Premiums):
        self.PremiumFuncs_init = self.PremiumFuncs # So that these are used on the next iteration
        return PremiumFuncsContainer(self.PremiumFuncs)
        
        
    def combineESIandIMIpremiums(self,IMIpremiums,ESIpremiums):
        '''
        Combine a 1D array of ESI premiums with a 3D array of IMIpremiums
        (age,health,contract) to generate a triple nested list of PremiumFuncs.
        
        Parameters
        ----------
        IMIpremiums : np.array
            3D array of individual market insurance premiums, organized as (age,health,contract).
        ESIpremiums : np.array
            1D array of employer sponsored insurance premiums, assumed to be constant
            across age and health status.  First element should always be zero.
        
        Returns
        -------
        CombinedPremiumFuncs : PremiumFuncsContainer
            Single object with a triple nested list of PremiumFuncs.
        '''
        HealthCount = IMIpremiums.shape[1] # This should always be 5
        ESIcontractCount = ESIpremiums.size
        IMIcontractCount = IMIpremiums.shape[2]
        PremiumFuncs_all = []
        
        # Construct PremiumsFuncs for ESI contracts
        ESIpremiumFuncs = []
        for z in range(ESIcontractCount):
            ESIpremiumFuncs.append(ConstantFunction(ESIpremiums[z]))
        ESIpremiumFuncs_all_health = 2*HealthCount*[ESIpremiumFuncs]
        
        # Construct PremiumFuncs for IMI contracts and combine with IMI contracts
        for t in range(40):
            PremiumFuncs_t = []
            for h in range(HealthCount):
                PremiumFuncsIMI = []
                for z in range(IMIcontractCount):
                    PremiumFuncsIMI.append(ConstantFunction(IMIpremiums[t,h,z]))
                PremiumFuncs_t.append(PremiumFuncsIMI)
            PremiumFuncs_t += ESIpremiumFuncs_all_health # Add on ESI premiums to end
            PremiumFuncs_all.append(PremiumFuncs_t)
            
        # Add on retired premiums, which are trivial
        RetPremiumFuncs = [ConstantFunction(0.0)]
        RetPremiumFuncs_all_health = HealthCount*[RetPremiumFuncs]
        for t in range(20):
            PremiumFuncs_all.append(RetPremiumFuncs_all_health)
            
        # Package the PremiumFuncs into a single object and return it
        CombinedPremiumFuncs = PremiumFuncsContainer(PremiumFuncs_all)
        return CombinedPremiumFuncs
        

    def ESIactuarialRule(self, ExpInsPay, ExpBuyers):
        '''
        Constructs a nested list of premium functions that have a constant value,
        based on the average medical needs of all contract buyers in the ESI market.
        
        Parameters
        ----------
        ExpInsPay : np.array
            4D array of expected insurance benefits paid by age-mrkv-contract-type.
        ExpBuyers : np.array
            4D array of expected contract buyers by age-mrkv-contract-type.
        
        Returns
        -------
        ESIpremiums np.array
            Vector with newly calculated ESI premiums; first element is always zero.
        '''
        ExpInsPayX = np.stack(ExpInsPay,axis=3)[:40,5:,:,:] # This is collected in reap_vars
        ExpBuyersX = np.stack(ExpBuyers,axis=3)[:40,5:,:,:] # This is collected in reap_vars
        # Order of indices: age, health, contract, type
        
        TotalInsPay = np.sum(ExpInsPayX,axis=(0,1,3))
        TotalBuyers = np.sum(ExpBuyersX,axis=(0,1,3))
        AvgInsPay   = TotalInsPay/TotalBuyers
        
        DampingFac = 0.2
        try:
            ESIpremiums = (1.0-DampingFac)*self.LoadFacESI*AvgInsPay + DampingFac*self.ESIpremiums
        except:
            ESIpremiums = self.LoadFacESI*AvgInsPay
        ESIpremiums[0] = 0.0 # First contract is always free
        
        print('ESI premiums: ' + str(ESIpremiums[1]) + ', insured rate: ' + str(TotalBuyers[1]/np.sum(TotalBuyers)))
        return ESIpremiums
        

###############################################################################
## Define various individual market insurance actuarial rules #################      
###############################################################################

def flatActuarialRule(self,ExpInsPay,ExpBuyers):
    '''
    Constructs a nested list of premium functions that have a constant value,
    based on the average medical needs of all contract buyers in the IMI market.
    This emulates how the ESI market works.
    
    Parameters
    ----------
    ExpInsPay : np.array
        4D array of expected insurance benefits paid by age-health-contract-type.
    ExpBuyers : np.array
        4D array of expected contract buyers by age-health-contract-type.
    
    Returns
    -------
    IMIpremiumArray : np.array
        3D array with individual market insurance premiums, ordered (age,health,contract).
    '''
    HealthCount = ExpInsPay[0].shape[1]
    MaxContracts = ExpInsPay[0].shape[2]
    ExpInsPayX = np.stack(ExpInsPay,axis=3)[:40,:5,:,:] # This is collected in reap_vars
    ExpBuyersX = np.stack(ExpBuyers,axis=3)[:40,:5,:,:] # This is collected in reap_vars
    # Order of indices: age, health, contract, type
    
    TotalInsPay = np.sum(ExpInsPayX,axis=(0,1,3))
    TotalBuyers = np.sum(ExpBuyersX,axis=(0,1,3))
    AvgInsPay   = TotalInsPay/TotalBuyers
    DampingFac = 0.2
    try:
        IMIpremiums = (1.0-DampingFac)*self.LoadFacIMI*AvgInsPay + DampingFac*self.IMIpremiums
    except:
        IMIpremiums = self.LoadFacIMI*AvgInsPay
    IMIpremiums[0] = 0.0 # First contract is always free
    print('IMI premiums: ' + str(IMIpremiums[1]) + ', insured rate: ' + str(TotalBuyers[1]/np.sum(TotalBuyers)))
    
    IMIpremiumArray = np.tile(np.reshape(IMIpremiums,(1,1,MaxContracts)),(40,HealthCount,1))
    return IMIpremiumArray


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
    try:
        NewPremiums = (1.0-DampingFac)*self.LoadFac*AvgInsPay + DampingFac*self.Premiums
    except:
        NewPremiums = self.LoadFac*AvgInsPay    
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
    HealthStateGroups = self.HealthStateGroups
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
    #HealthStateGroups = [[0,1],[2,3,4]]
    #HealthStateGroups = [[0,1,2,3,4]]
    HealthStateGroups = self.HealthStateGroups
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
    AgeBandLimit = self.AgeBandLimit
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
    
    AgeRatingScale = 1.0 + (AgeBandLimit-1.0)*AgeRatingFunc(np.arange(AgeCount,dtype=float))
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
    
