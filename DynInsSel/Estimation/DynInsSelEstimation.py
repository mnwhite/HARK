'''
This is a first attempt at the DynInsSel estimation.  It has functions to construct agent type(s)
from input parameters; it will eventually have the ability to compare simulated data to real data.
'''
import numpy as np
import DynInsSelParams as Params
from copy import copy
from InsuranceSelectionModel import MedInsuranceContract, InsSelConsumerType
from HARKinterpolation import ConstantFunction

def makeDynInsSelType(CRRAcon,CRRAmed,DiscFac,ChoiceShkMag,MedShkMeanAgeParams,MedShkMeanVGparams,
                      MedShkMeanGDparams,MedShkMeanFRparams,MedShkMeanPRparams,MedShkStdAgeParams,
                      MedShkStdVGparams,MedShkStdGDparams,MedShkStdFRparams,MedShkStdPRparams,
                      PremiumArray,PremiumSubsidy):
                          
    # Make a dictionary as a copy from the parameter file (already has most parameters)
    TypeDict = copy(Params.BasicDictionary)
    TypeDict['CRRA'] = CRRAcon
    TypeDict['CRRAmed'] = CRRAmed
    TypeDict['DiscFac'] = DiscFac
    TypeDict['ChoiceShkMag'] = ChoiceShkMag
                          
    # Make arrays of medical shock means and standard deviations
    MedShkMeanArray = np.array(5,Params.AgeCount)
    MedShkStdArray  = np.array(5,Params.AgeCount)
    AgeArray = np.arange(Params.AgeCount,dtype=float)
    MedShkMeanEXfunc = lambda age : MedShkMeanAgeParams[0] + MedShkMeanAgeParams[1]*age + MedShkMeanAgeParams[2]*age**2 \
                        + MedShkMeanAgeParams[3]*age**3 + MedShkMeanAgeParams[4]*age**4
    MedShkMeanVGfunc = lambda age : MedShkMeanEXfunc(age) + MedShkMeanVGparams[0] + MedShkMeanVGparams[1]*age
    MedShkMeanGDfunc = lambda age : MedShkMeanVGfunc(age) + MedShkMeanGDparams[0] + MedShkMeanGDparams[1]*age
    MedShkMeanFRfunc = lambda age : MedShkMeanGDfunc(age) + MedShkMeanFRparams[0] + MedShkMeanFRparams[1]*age
    MedShkMeanPRfunc = lambda age : MedShkMeanFRfunc(age) + MedShkMeanPRparams[0] + MedShkMeanPRparams[1]*age
    MedShkStdEXfunc = lambda age : MedShkStdAgeParams[0] + MedShkStdAgeParams[1]*age + MedShkStdAgeParams[2]*age**2 \
                        + MedShkStdAgeParams[3]*age**3 + MedShkStdAgeParams[4]*age**4
    MedShkStdVGfunc = lambda age : MedShkStdEXfunc(age) + MedShkStdVGparams[0] + MedShkStdVGparams[1]*age
    MedShkStdGDfunc = lambda age : MedShkStdVGfunc(age) + MedShkStdGDparams[0] + MedShkStdGDparams[1]*age
    MedShkStdFRfunc = lambda age : MedShkStdGDfunc(age) + MedShkStdFRparams[0] + MedShkStdFRparams[1]*age
    MedShkStdPRfunc = lambda age : MedShkStdFRfunc(age) + MedShkStdPRparams[0] + MedShkStdPRparams[1]*age
    MedShkMeanArray[4,:] = np.exp(MedShkMeanEXfunc(AgeArray))
    MedShkMeanArray[3,:] = np.exp(MedShkMeanVGfunc(AgeArray))
    MedShkMeanArray[2,:] = np.exp(MedShkMeanGDfunc(AgeArray))
    MedShkMeanArray[1,:] = np.exp(MedShkMeanFRfunc(AgeArray))
    MedShkMeanArray[0,:] = np.exp(MedShkMeanPRfunc(AgeArray))
    MedShkStdArray[4,:] = np.exp(MedShkStdEXfunc(AgeArray))
    MedShkStdArray[3,:] = np.exp(MedShkStdVGfunc(AgeArray))
    MedShkStdArray[2,:] = np.exp(MedShkStdGDfunc(AgeArray))
    MedShkStdArray[1,:] = np.exp(MedShkStdFRfunc(AgeArray))
    MedShkStdArray[0,:] = np.exp(MedShkStdPRfunc(AgeArray))
    TypeDict['MedShkMean'] = MedShkMeanArray.tolist()
    TypeDict['MedShkStd'] = MedShkStdArray.tolist()
    
    # Make insurance contracts when working and retired, then combine into a lifecycle list
    WorkingContractList = []
    WorkingContractList.append(MedInsuranceContract(ConstantFunction(0.0),0.0,1.0,Params.MedPrice))
    for j in range(PremiumArray.size):
        Premium = max([PremiumArray[j] - PremiumSubsidy,0.0])
        Copay = 0.05
        Deductible = Params.DeductibleList[j]
        WorkingContractList.append(MedInsuranceContract(ConstantFunction(Premium),Deductible,Copay,Params.MedPrice))
    RetiredContractList = [MedInsuranceContract(ConstantFunction(0.0),0.0,0.1,Params.MedPrice)]
    TypeDict['ContractList'] = Params.T_working*[5*[WorkingContractList]] + Params.T_retired*[5*[RetiredContractList]]
    
    # Make and return a DynInsSelType
    ThisType = InsSelConsumerType(TypeDict)
    return ThisType
        
    