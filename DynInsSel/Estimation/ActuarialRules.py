'''
This module contains various "actuarial rules" for calculating premium functions
from the medical costs of consumers.
'''
import numpy as np
from HARKcore import HARKobject, Market
from HARKinterpolation import ConstantFunction
from scipy.optimize import brentq
import matplotlib.pyplot as plt
from SubsidyFuncs import NullSubsidyFuncs, makeACAstyleSubsidyPolicy

# This is a trivial "container" class
class PremiumFuncsContainer(HARKobject):
    distance_criteria = ['PremiumFuncs']
    
    def __init__(self,PremiumFuncs,HealthTaxRate=None):
        self.PremiumFuncs = PremiumFuncs
        if HealthTaxRate is not None:
            self.HealthTaxRate = HealthTaxRate
            

# This is also a trivial "container" class
class PolicySpecification(object):
    '''
    A very simple class for representing specifications for counterfactuals.
    Attributes include the IMI subsidy function, an IMI ActuarialRule function,
    HealthGroups, AgeBandLimit, an individual MandateTaxRate and Floor, and a name.
    Each of these attributes is added as attributes of the InsuranceMarket.
    '''
    def __init__(self,SubsidyFunc,ActuarialRule,HealthGroups,ExcludedGroups,AgeBandLimit,
                 MandateTaxRate,MandateFloor,MandateForESI,name,text):
        self.SubsidyFunc   = SubsidyFunc
        self.ActuarialRule = ActuarialRule
        self.HealthGroups  = HealthGroups
        self.ExcludedGroups= ExcludedGroups
        self.AgeBandLimit  = AgeBandLimit
        self.MandateTaxRate= MandateTaxRate
        self.MandateFloor  = MandateFloor
        self.MandateForESI = MandateForESI
        self.name          = name
        self.text          = text
        
        
class InsuranceMarket(Market):
    '''
    A class for representing the "insurance economy" with many agent types.
    '''

    def __init__(self,PolicySpec,Agents):
        Market.__init__(self,agents=[],sow_vars=['PremiumFuncs'],
                        reap_vars=['ExpInsPay','ExpBuyers','IncAboveThreshByAge','HealthBudgetByAge'],
                        const_vars=[],
                        track_vars=['ESIpremiums','IMIpremiums'],
                        dyn_vars=['PremiumFuncs','HealthTaxRate'],
                        millRule=None,calcDynamics=None,act_T=10,tolerance=0.0001)
        self.agents = Agents
        self.updatePolicy(PolicySpec)
        
        
    def updatePolicy(self,PolicySpec):
        self.IMIactuarialRule = PolicySpec.ActuarialRule
        self.SubsidyFunc      = PolicySpec.SubsidyFunc
        self.HealthGroups     = PolicySpec.HealthGroups
        self.ExcludedGroups   = PolicySpec.ExcludedGroups
        self.AgeBandLimit     = PolicySpec.AgeBandLimit
        self.MandateTaxRate   = PolicySpec.MandateTaxRate
        self.MandateFloor     = PolicySpec.MandateFloor
        self.MandateForESI    = PolicySpec.MandateForESI
        self.applySubsidyAndIM()
        
    
    def applySubsidyAndIM(self):
        # Have each agent type in the market inherit the IMI subsidies and individual mandate
        for this_type in self.agents:
            setattr(this_type,'SubsidyFunc', self.SubsidyFunc)
            this_type.updateUninsuredPremium(self.MandateTaxRate, self.MandateFloor, self.MandateForESI)
            
    
    def millRule(self,ExpInsPay,ExpBuyers,IncAboveThreshByAge,HealthBudgetByAge):
        IMIpremiums      = self.IMIactuarialRule(self,ExpInsPay,ExpBuyers)
        ESIpremiums      = self.ESIactuarialRule(ExpInsPay,ExpBuyers)
        self.IncAboveThreshByAge = IncAboveThreshByAge
        self.HealthBudgetByAge = HealthBudgetByAge
        PremiumFuncs = self.combineESIandIMIpremiums(IMIpremiums,ESIpremiums)
        return PremiumFuncs
        
    
    def calcDynamics(self):
        self.PremiumFuncs_init = self.PremiumFuncs # So that these are used on the next iteration
        HealthTaxRate = self.calcHealthTaxRate()
        return PremiumFuncsContainer(self.PremiumFuncs,HealthTaxRate)
        
        
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
        MaxContracts = ExpInsPayX.shape[2]
        # Order of indices: age, health, contract, type
        
        TotalInsPay_ByAge = np.sum(ExpInsPayX,axis=(1,3))
        TotalBuyers_ByAge = np.sum(ExpBuyersX,axis=(1,3))
        CohortWeight = self.CohortGroFac**(-np.arange(40))
        CohortWeightX = np.tile(np.reshape(CohortWeight,(40,1)),(1,MaxContracts))
        TotalInsPay = np.sum(CohortWeightX*TotalInsPay_ByAge,axis=0)
        TotalBuyers = np.sum(CohortWeightX*TotalBuyers_ByAge,axis=0)
        AvgInsPay = TotalInsPay/TotalBuyers
        
        DampingFac = 0.0
        try:
            ESIpremiums = (1.0-DampingFac)*self.LoadFacESI*AvgInsPay + DampingFac*self.ESIpremiums
        except:
            ESIpremiums = self.LoadFacESI*AvgInsPay
        ESIpremiums[0] = 0.0 # First contract is always free
        self.ESIpremiums = ESIpremiums
        
        print('ESI premiums: ' + str(ESIpremiums[1]) + ', insured rate: ' + str(TotalBuyers[1]/np.sum(TotalBuyers)))
        return ESIpremiums
    
    
    def calcHealthTaxRate(self):
        '''
        Calculate the tax rate that would pay for the health spending activities
        in the economy: Cfloor welfare, Medicare, insurance subsidies, (minus) IM penalties.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        HealthTaxRate : float
            New tax rate that would pay for current levels of health spending.
        '''
        IncAboveThreshByAge = np.sum(np.stack(self.IncAboveThreshByAge,axis=1),axis=1)
        HealthBudgetByAge = np.sum(np.stack(self.HealthBudgetByAge,axis=1),axis=1)
        AgeCount = IncAboveThreshByAge.size
        
        WeightVec = self.CohortGroFac**(-np.arange(AgeCount))
        IncAboveThresh = np.dot(WeightVec,IncAboveThreshByAge)
        HealthBudget = np.dot(WeightVec,HealthBudgetByAge)
        
        HealthTaxRate = HealthBudget/IncAboveThresh
        print('New health tax rate is '+ str(HealthTaxRate))
        return HealthTaxRate
        
        

###############################################################################
## Define an individual market insurance actuarial rule #################      
###############################################################################


def generalIMIactuarialRule(self,ExpInsPay,ExpBuyers):
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
    AgeBandLimit = self.AgeBandLimit
    HealthGroups = self.HealthGroups
    ExcludedGroups = self.ExcludedGroups
    GroupCount = len(HealthGroups)
    AgeCount = 40
    InfPrem = 10000.
    
    # Define the age rating function
    AgeRatingFunc = lambda x : (np.exp(x/40.*3.0)-1.)/(np.exp(3.0)-1.)
    if AgeBandLimit is not None:
        AgeRatingScale = 1.0 + (AgeBandLimit-1.0)*AgeRatingFunc(np.arange(AgeCount,dtype=float))
    
    # Combine expected payments and buyers across types
    ExpInsPayX = np.stack(ExpInsPay,axis=3)[:40,:5,:,:] # This is collected in reap_vars
    ExpBuyersX = np.stack(ExpBuyers,axis=3)[:40,:5,:,:] # This is collected in reap_vars
    HealthCount = ExpInsPayX.shape[1]
    MaxContracts = ExpInsPayX.shape[2]
    CohortWeight = self.CohortGroFac**(-np.arange(40))
    CohortWeightX = np.tile(np.reshape(CohortWeight,(40,1)),(1,MaxContracts))
    # Order of indices: age, health, contract, type
    
    # Initialize the premium array
    PremiumArray = np.zeros((AgeCount,GroupCount,MaxContracts))
    
    # Loop through each health group and calculate new premiums for each contract
    for g in range(GroupCount):
        these = HealthGroups[g] # Health levels for this group
        TotalInsPayByAge = np.sum(ExpInsPayX[:,these,:,:],axis=(1,3)) # Don't sum across ages
        TotalBuyersByAge = np.sum(ExpBuyersX[:,these,:,:],axis=(1,3)) # Don't sum across ages
        TotalInsPayByAge *= CohortWeightX # Cohort-size adjusted
        TotalBuyersByAge *= CohortWeightX # Cohort-size adjusted 
        
        if AgeBandLimit is not None: # If there is an age band limit...
            # Loop through each contract and zero out the profit function
            for z in range(1,MaxContracts):
                TotalCost = np.sum(TotalBuyersByAge[:,z]*0.12 + TotalInsPayByAge[:,z]*self.LoadFacIMI)
                
                def tempFunc(BasePremium): # Profit function to be zero'ed
                    PremiumVec = BasePremium*AgeRatingScale
                    TotalRevenue = np.sum(PremiumVec*TotalBuyersByAge[:,z])
                    return TotalRevenue - TotalCost
                
                NewBasePremium = brentq(tempFunc, 0.0, 1.0)
                PremiumArray[:,g,z] = NewBasePremium*AgeRatingScale
        
        else: # If there is not an age band limit...
            AvgInsPayByAge = TotalInsPayByAge/TotalBuyersByAge
            NewPremiums = self.LoadFacIMI*AvgInsPayByAge + 0.12
            PremiumArray[:,g,:] = NewPremiums
            PremiumArray[:,g,0] = 0.
            
        if ExcludedGroups[g]: # Exclude group with absurdly high premium
            PremiumArray[:,g,1:] = InfPrem
    
    self.IMIpremiums = PremiumArray
    
    # Calculate and report overall insured rate
    TotalBuyersByAge = np.sum(ExpBuyersX,axis=(1,3))
    TotalBuyersByAge *= CohortWeightX # Cohort-size adjusted
    TotalBuyers = np.sum(TotalBuyersByAge,axis=0)
    print('IMI insured rate: ' + str(TotalBuyers[1]/np.sum(TotalBuyers)))
    plt.plot(PremiumArray[:,:,1])
    plt.show()
    
    # Distribute health group premiums into each health state as appropriate
    IMIpremiums = np.zeros((AgeCount,HealthCount,MaxContracts))
    for g in range(GroupCount):
        for h in HealthGroups[g]:
            IMIpremiums[:,h,:] = PremiumArray[:,g,:]
    return IMIpremiums

    

# Specify the baseline structure of the insurance market
BaselinePolicySpec = PolicySpecification(SubsidyFunc=NullSubsidyFuncs,
                                         ActuarialRule = generalIMIactuarialRule,
                                         HealthGroups = [[0,1],[2,3,4]],
                                         ExcludedGroups = [False,False],
                                         AgeBandLimit = None,
                                         MandateTaxRate = 0.00,
                                         MandateFloor = 0.00,
                                         MandateForESI = False,
                                         name = 'Baseline',
                                         text = 'baseline specification')
        
        