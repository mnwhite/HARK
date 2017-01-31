'''
Example/testing parameters for the insurance selection model.
'''
import numpy as np

CRRA = 2.0                          # Coefficient of relative risk aversion (for consumption)
CRRAmed = 1.5*CRRA                  # Coefficient of relative risk aversion (for medical care)
Rfree = 5*[1.03]                    # Interest factor on assets
DiscFac = 0.96                      # Intertemporal discount factor
LivPrb = [[0.96,0.97,0.98,0.99,1.00]]# Survival probability by discrete health state
#LivPrb = [[0.98,0.98,0.98,0.98,0.98]]
PermGroFac = [[0.98,0.99,1.0,1.01,1.02]]# Permanent income growth factor by discrete health state
#PermGroFac = [[1.0,1.0,1.0,1.0,1.0]]
aXtraMin = 0.001                    # Minimum end-of-period "assets above minimum" value
aXtraMax = 80                       # Minimum end-of-period "assets above minimum" value               
aXtraExtra = [0.005,0.01]           # Some other value of "assets above minimum" to add to the grid, not used
aXtraNestFac = 3                    # Exponential nesting factor when constructing "assets above minimum" grid
aXtraCount = 32                    # Number of points in the grid of "assets above minimum"
PermShkCount = 5                    # Number of points in discrete approximation to permanent income shocks
TranShkCount = 5                    # Number of points in discrete approximation to transitory income shocks
PermShkStd = [[0.2,0.15,0.1,0.1,0.1]] # Standard deviation of log permanent income shocks
TranShkStd = [[0.2,0.15,0.1,0.1,0.1]] # Standard deviation of log transitory income shocks
#PermShkStd = [[0.1,0.1,0.1,0.1,0.1]]
#TranShkStd = [[0.1,0.1,0.1,0.1,0.1]]
UnempPrb = 0.05                     # Probability of unemployment while working
UnempPrbRet = 0.005                 # Probability of "unemployment" while retired
IncUnemp = 0.3                      # Unemployment benefits replacement rate
IncUnempRet = 0.0                   # "Unemployment" benefits when retired
T_retire = 0                        # Period of retirement (0 --> no retirement)
BoroCnstArt = 0.0                   # Artificial borrowing constraint; imposed minimum level of end-of period assets
CubicBool = False                   # Use cubic spline interpolation when True, linear interpolation when False
PermIncCount = 12                   # Number of permanent income gridpoints in "body"
PermInc_tail_N = 3                  # Number of permanent income gridpoints in each "tail"
PermIncStdInit = 0.4                # Initial standard deviation of (log) permanent income (not used in example)
PermIncAvgInit = 1.0                # Initial average of permanent income (not used in example)
PermIncCorr = 1.0                   # Serial correlation coefficient for permanent income
MedShkAvg = [[2.0,1.6,1.2,0.8,0.4]] # Average of (log) medical need shocks
MedShkStd = [[5.0,5.0,5.0,5.0,5.0]] # Standard deviation of (log) medical need shocks
MedShkCount = 3                     # Number of medical shock points in "body"
MedShkCountTail = 10                # Number of medical shock points in "tail" (upper only)
MedPrice = [1.5]                    # Relative price of a unit of medical care
MrkvArray = [np.array([[0.80,0.10,0.05,0.03,0.02], # Markov transition array between health states
                      [0.10,0.70,0.10,0.07,0.03],
                      [0.05,0.10,0.70,0.10,0.05],
                      [0.03,0.07,0.10,0.70,0.10],
                      [0.01,0.03,0.06,0.20,0.70]])]
ChoiceShkMag = 0.01                # Magnitude of choice shocks (over insurance contracts)
T_cycle = 10                       # Total number of periods in cycle for this agent
AgentCount = 10000                 # Number of agents of this type (only matters for simulation)

init_insurance_selection = { 'CRRA': CRRA,
                             'Rfree': Rfree,
                             'DiscFac': DiscFac,
                             'LivPrb': T_cycle*LivPrb,
                             'PermGroFac': T_cycle*PermGroFac,
                             'CRRAmed': CRRAmed,
                             'aXtraMin': aXtraMin,
                             'aXtraMax': aXtraMax,
                             'aXtraExtra': aXtraExtra,
                             'aXtraNestFac': aXtraNestFac,
                             'aXtraCount': aXtraCount,
                             'PermShkCount': PermShkCount,
                             'TranShkCount': TranShkCount,
                             'PermShkStd': T_cycle*PermShkStd,
                             'TranShkStd': T_cycle*TranShkStd,
                             'UnempPrb': UnempPrb,
                             'UnempPrbRet': UnempPrbRet,
                             'IncUnemp': IncUnemp,
                             'IncUnempRet': IncUnempRet,
                             'T_retire': T_retire,
                             'BoroCnstArt': BoroCnstArt,
                             'CubicBool': CubicBool,
                             'PermIncCount': PermIncCount,
                             'PermInc_tail_N': PermInc_tail_N,
                             'PermIncStdInit': PermIncStdInit,
                             'PermIncAvgInit': PermIncAvgInit,
                             'PermIncCorr': PermIncCorr,
                             'MedShkAvg': T_cycle*MedShkAvg,
                             'MedShkStd': T_cycle*MedShkStd,
                             'MedShkCount': MedShkCount,
                             'MedShkCountTail': MedShkCountTail,
                             'MedPrice': T_cycle*MedPrice,
                             'MrkvArray': T_cycle*MrkvArray,
                             'ChoiceShkMag': T_cycle*[ChoiceShkMag],
                             'T_cycle': T_cycle,
                             'AgentCount': AgentCount
                            }