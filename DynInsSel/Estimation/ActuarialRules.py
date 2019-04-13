'''
This module contains various "actuarial rules" for calculating premium functions
from the medical costs of consumers.
'''
import numpy as np
from HARKcore import HARKobject, Market
from HARKinterpolation import ConstantFunction
from copy import copy, deepcopy
from scipy.optimize import brentq
import matplotlib.pyplot as plt

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
                        track_vars=['ESIpremiums','IMIpremiums'],
                        dyn_vars=['PremiumFuncs'],
                        millRule=None,calcDynamics=None,act_T=10,tolerance=0.0001)
        self.IMIactuarialRule = ActuarialRule

    def millRule(self,ExpInsPay,ExpBuyers):
        IMIpremiums      = self.IMIactuarialRule(self,ExpInsPay,ExpBuyers)
        ESIpremiums      = self.ESIactuarialRule(ExpInsPay,ExpBuyers)
        self.IMIpremiums = IMIpremiums
        self.ESIpremiums = ESIpremiums
        return self.combineESIandIMIpremiums(IMIpremiums,ESIpremiums)
        
    def calcDynamics(self):
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
            
#        # Add on retired premiums, which are trivial
#        RetPremiumFuncs = [ConstantFunction(0.0)]
#        RetPremiumFuncs_all_health = HealthCount*[RetPremiumFuncs]
#        for t in range(20):
#            PremiumFuncs_all.append(RetPremiumFuncs_all_health)
            
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
    ExpInsPayX = np.stack(ExpInsPay,axis=3)[:40,:5,:,:] # This is collected in reap_vars
    ExpBuyersX = np.stack(ExpBuyers,axis=3)[:40,:5,:,:] # This is collected in reap_vars
    HealthCount = ExpInsPayX.shape[1]
    MaxContracts = ExpInsPayX.shape[2]
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
    self.IMIpremiums = IMIpremiums
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
    ExpInsPay : np.array
        4D array of expected insurance benefits paid by age-health-contract-type.
    ExpBuyers : np.array
        4D array of expected contract buyers by age-health-contract-type.
    
    Returns
    -------
    IMIpremiumArray : np.array
        3D array with individual market insurance premiums, ordered (age,health,contract).
    '''
    ExcludedHealth = self.ExcludedHealth
    InfPrem = 10000.0
    
    ExpInsPayX = np.stack(ExpInsPay,axis=3)[:40,:5,:,:] # This is collected in reap_vars
    ExpBuyersX = np.stack(ExpBuyers,axis=3)[:40,:5,:,:] # This is collected in reap_vars
    HealthCount = ExpInsPayX.shape[1]
    MaxContracts = ExpInsPayX.shape[2]
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
    self.IMIpremiums = IMIpremiums
    print('IMI premiums: ' + str(IMIpremiums[1]) + ', insured rate: ' + str(TotalBuyers[1]/np.sum(TotalBuyers)))
    
    IMIpremiumArray = np.tile(np.reshape(IMIpremiums,(1,1,MaxContracts)),(40,HealthCount,1))
    IMIpremiumArray[:,ExcludedHealth,1:] = InfPrem
    return IMIpremiumArray

    
def healthRatedActuarialRule(self,ExpInsPay,ExpBuyers):
    '''
    Constructs a nested list of premium functions that have a constant value,
    based on the average medical needs of contract buyers in the same health
    state group; the groups are flexible.
    
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
    HealthGroups = self.HealthGroups
    GroupCount = len(HealthGroups)
    
    ExpInsPayX = np.stack(ExpInsPay,axis=3)[:40,:5,:,:] # This is collected in reap_vars
    ExpBuyersX = np.stack(ExpBuyers,axis=3)[:40,:5,:,:] # This is collected in reap_vars
    HealthCount = ExpInsPayX.shape[1]
    MaxContracts = ExpInsPayX.shape[2]
    # Order of indices: age, health, contract, type
    
    PremiumArray = np.zeros((GroupCount,MaxContracts)) + np.nan
    for g in range(GroupCount):
        these = HealthGroups[g]
        TotalInsPay = np.sum(ExpInsPayX[:,these,:,:],axis=(0,1,3))
        TotalBuyers = np.sum(ExpBuyersX[:,these,:,:],axis=(0,1,3))
        AvgInsPay   = TotalInsPay/TotalBuyers
        DampingFac = 0.2
        try:
            NewPremiums = (1.0-DampingFac)*self.LoadFacIMI*AvgInsPay + DampingFac*self.IMIpremiums[:,g]
        except:
            NewPremiums = self.LoadFacIMI*AvgInsPay
        NewPremiums[0] = 0.0 # First contract is always free
        PremiumArray[g,:] = NewPremiums
        print('IMI premiums group ' + str(g) + ': ' + str(NewPremiums[1]) + ', insured rate: ' + str(TotalBuyers[1]/np.sum(TotalBuyers)))
    self.IMIpremiums = PremiumArray

    IMIpremiums = np.zeros((40,HealthCount,MaxContracts))
    for g in range(GroupCount):
        for h in HealthGroups[g]:
            for t in range(40):
                IMIpremiums[t,h,:] = PremiumArray[g,:]
    return IMIpremiums
    

def ageHealthRatedActuarialRule(self,ExpInsPay,ExpBuyers):
    '''
    Constructs a nested list of premium functions that have a constant value,
    based on the average medical needs of contract buyers in the same health
    state group and age; the health groups are defined flexibly.  This is the
    "before ACA" baseline for the individual market.
    
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
    HealthGroups = self.HealthGroups
    ExcludedGroups = self.ExcludedGroups
    GroupCount = len(HealthGroups)
    AgeCount = 40
    InfPrem = 10000.
    
    ExpInsPayX = np.stack(ExpInsPay,axis=3)[:40,:5,:,:] # This is collected in reap_vars
    ExpBuyersX = np.stack(ExpBuyers,axis=3)[:40,:5,:,:] # This is collected in reap_vars
    HealthCount = ExpInsPayX.shape[1]
    MaxContracts = ExpInsPayX.shape[2]
    # Order of indices: age, health, contract, type
    
    PremiumArray = np.zeros((AgeCount,GroupCount,MaxContracts)) + np.nan
    for g in range(GroupCount):
        these = HealthGroups[g]
        TotalInsPay = np.sum(ExpInsPayX[:,these,:,:],axis=(1,3)) # Don't sum across ages
        TotalBuyers = np.sum(ExpBuyersX[:,these,:,:],axis=(1,3)) # Don't sum across ages
        AvgInsPay   = TotalInsPay/TotalBuyers
        DampingFac = 0.2
        try:
            NewPremiums = (1.0-DampingFac)*self.LoadFacIMI*AvgInsPay + DampingFac*self.IMIpremiums[:,g,:]
        except:
            NewPremiums = self.LoadFacIMI*AvgInsPay
        NewPremiums[:,0] = 0.0 # First contract is always free
        PremiumArray[:,g,:] = NewPremiums
    self.IMIpremiums = PremiumArray
    
    TotalBuyers = np.sum(ExpBuyersX[:,these,:,:],axis=(0,1,3))
    print('IMI insured rate: ' + str(TotalBuyers[1]/np.sum(TotalBuyers)))
    #plt.plot(PremiumArray[:,:,1])
    #plt.show()
        
    IMIpremiums = np.zeros((AgeCount,HealthCount,MaxContracts))
    for g in range(GroupCount):
        for h in HealthGroups[g]:
            if ExcludedGroups[g]:
                IMIpremiums[:,h,1:] = InfPrem
            else:
                IMIpremiums[:,h,:] = PremiumArray[:,g,:]
    return IMIpremiums
    

def ageRatedActuarialRule(self,ExpInsPay,ExpBuyers):
    '''
    Constructs a nested list of premium functions that have a constant value,
    based on the average medical needs of contract buyers of the same age.
    An age-rating function is specified, which maps from the range of ages to
    the [0,1] interval.
    
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
    AgeRatingFunc = lambda x : (np.exp(x/40.*3.0)-1.)/(np.exp(3.0)-1.)
    AgeBandLimit = self.AgeBandLimit
    AgeCount = 40
    
    ExpInsPayX = np.stack(ExpInsPay,axis=3)[:40,:5,:,:] # This is collected in reap_vars
    ExpBuyersX = np.stack(ExpBuyers,axis=3)[:40,:5,:,:] # This is collected in reap_vars
    HealthCount = ExpInsPayX.shape[1]
    MaxContracts = ExpInsPayX.shape[2]
    # Order of indices: age, health, contract, type
    
    PremiumArray = np.zeros((AgeCount,MaxContracts)) + np.nan
    TotalInsPay = np.sum(ExpInsPayX,axis=(0,1,3))
    TotalBuyers = np.sum(ExpBuyersX,axis=(0,1,3))
    TotalBuyersByAge = np.sum(ExpBuyersX[0:40,:,:,:],axis=(1,3)) # Don't sum across ages
    AvgInsPay = TotalInsPay/TotalBuyers
    
    AgeRatingScale = 1.0 + (AgeBandLimit-1.0)*AgeRatingFunc(np.arange(AgeCount,dtype=float))
    for z in range(1,MaxContracts):
        def tempFunc(BasePremium):
            PremiumVec = BasePremium*AgeRatingScale
            TotalRevenue = np.sum(PremiumVec*TotalBuyersByAge[:,z])
            return TotalRevenue/TotalInsPay[z] - self.LoadFacIMI
        
        NewPremium = brentq(tempFunc,0.0,AvgInsPay[z]*self.LoadFacIMI)
        PremiumArray[:,z] = NewPremium*AgeRatingScale
    self.IMIpremiums = PremiumArray
    
    print('IMI insured rate: ' + str(TotalBuyers[1]/np.sum(TotalBuyers)))
    #plt.plot(PremiumArray[:,1])
    #plt.show()
    
    IMIpremiums = np.tile(np.reshape(PremiumArray,(AgeCount,1,MaxContracts)),(1,HealthCount,1))
    return IMIpremiums
    
