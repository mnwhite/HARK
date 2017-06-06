'''
This module runs a quick and dirty structural estimation based on Table 9 of 
"MPC Heterogeneity and Household Balance Sheets" by Fagereng, Holm, and Natvik.
'''

import sys 
import os
sys.path.insert(0, os.path.abspath('../'))
sys.path.insert(0, os.path.abspath('../ConsumptionSaving'))
sys.path.insert(0, os.path.abspath('../cstwMPC'))

import numpy as np
import matplotlib.pyplot as plt
from copy import copy, deepcopy
from time import clock

from HARKutilities import approxUniform, getPercentiles
from HARKparallel import multiThreadCommands
from ConsIndShockModel import IndShockConsumerType
from SetupParamsCSTWnew import init_infinite # dictionary with most ConsumerType parameters

TypeCount = 8    # Number of consumer types with heterogeneous discount factors
AdjFactor = 1.0  # Factor by which to scale all of Fagereng's MPCs in Table 9
T_kill = 100     # Don't let agents live past this age

# Define the MPC targets from Table 9; element i,j is lottery quartile i, deposit quartile j
MPC_target_base = np.array([[1.047, 0.745, 0.720, 0.490],
                            [0.762, 0.640, 0.559, 0.437],
                            [0.663, 0.546, 0.390, 0.386],
                            [0.354, 0.325, 0.242, 0.216]])
MPC_target = AdjFactor*MPC_target_base

# Define the four lottery sizes, in thousands of USD; these are eyeballed centers/averages
lottery_size = np.array([1.625, 3.3741, 7.129, 20.0])

# Make an initialization dictionary on an annual basis
base_params = deepcopy(init_infinite)
base_params['LivPrb'] = [1.0 - (1.0 - init_infinite['LivPrb'][0])*4]
base_params['Rfree'] = 1.04/base_params['LivPrb'][0]
base_params['PermShkStd'] = [0.1]
base_params['TranShkStd'] = [0.1]
base_params['T_age'] = T_kill # Kill off agents if they manage to achieve T_kill working years
base_params['AgentCount'] = 10000
base_params['pLvlInitMean'] = np.log(23.72) # From Table 1, in USD
base_params['T_sim'] = T_kill # No point simulating past when agents would be killed off

# Make several consumer types to be used during estimation
BaseType = IndShockConsumerType(**base_params)
EstTypeList = []
for j in range(TypeCount):
    EstTypeList.append(deepcopy(BaseType))
    
# Define the objective function
def FagerengObjFunc(center,spread):
    '''
    Objective function for the quick and dirty structural estimation to fit
    Fagereng, Holm, and Natvik's Table 9 results with a basic infinite horizon
    consumption-saving model (with permanent and transitory income shocks).
    
    Parameters
    ----------
    center : float
        Center of the uniform distribution of discount factors.
    spread : float
        Width of the uniform distribution of discount factors.
        
    Returns
    -------
    distance : float
        Euclidean distance between simulated MPCs and (adjusted) Table 9 MPCs.
    '''
    # Give our consumer types the requested discount factor distribution
    beta_set = approxUniform(N=TypeCount,bot=center-spread,top=center+spread)[1]
    for j in range(TypeCount):
        EstTypeList[j](DiscFac = beta_set[j])
        
    # Solve and simulate all consumer types, then gather their wealth levels
    multiThreadCommands(EstTypeList,['solve()','initializeSim()','simulate()','unpackcFunc()'])
    WealthNow = np.concatenate([ThisType.aLvlNow for ThisType in EstTypeList])
    
    # Get wealth quartile cutoffs and distribute them to each consumer type
    quartile_cuts = getPercentiles(WealthNow,percentiles=[0.25,0.50,0.75])
    for ThisType in EstTypeList:
        WealthQ = np.zeros(ThisType.AgentCount)
        for n in range(3):
            WealthQ[ThisType.aLvlNow > quartile_cuts[n]] += 1
        ThisType(WealthQ = WealthQ)
        
    # Keep track of MPC sets in lists of lists of arrays
    MPC_set_list = [ [[],[],[],[]],
                     [[],[],[],[]],
                     [[],[],[],[]],
                     [[],[],[],[]] ]
    
    # Calculate the secant MPC for each of the four lottery sizes for all agents
    for ThisType in EstTypeList:
        ThisType.simulate(1)
        c_base = ThisType.cNrmNow
        MPC_secant = np.zeros((ThisType.AgentCount,4))
        for k in range(4): # Get secant MPC for all agents of this type
            Llvl = lottery_size[k]
            Lnrm = Llvl/ThisType.pLvlNow
            mAdj = ThisType.mNrmNow + Lnrm
            cAdj = ThisType.cFunc[0](mAdj)
            MPC_secant[:,k] = (cAdj - c_base)/Lnrm
            
        # Sort the secant MPCs into the proper MPC sets
        for q in range(4):
            these = ThisType.WealthQ == q
            for k in range(4):
                MPC_set_list[k][q].append(MPC_secant[these,k])
                
    # Calculate average within each MPC set
    simulated_MPC_means = np.zeros((4,4))
    for k in range(4):
        for q in range(4):
            MPC_array = np.concatenate(MPC_set_list[k][q])
            simulated_MPC_means[k,q] = np.mean(MPC_array)
            
    # Calculate Euclidean distance between simulated MPC averages and Table 9 targets
    distance = np.sqrt(np.sum((simulated_MPC_means - MPC_target)**2))
    #print(simulated_MPC_means)
    return distance


if __name__ == '__main__':
    
    t_start = clock()
    X = FagerengObjFunc(0.814,0.122)
    t_end = clock()
    print('That took ' + str(t_end - t_start) + ' seconds.')
    print(X)
    
#    test_vec = np.linspace(0.80,0.83,40)
#    out_vec = np.empty(test_vec.size)
#    for j in range(test_vec.size):
#        beta = test_vec[j]
#        out_vec[j] = FagerengObjFunc(beta,0.122)
#        
#    plt.plot(test_vec,out_vec)
#    plt.show()
    
    
    