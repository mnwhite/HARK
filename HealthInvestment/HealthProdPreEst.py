'''
This module runs a pre-estimation on the health production parameters, given
values of all other parameters.
'''

import sys
import os
sys.path.insert(0,'../')

from time import clock
from copy import copy
import numpy as np
from HARKsimulation import drawNormal
from HARKparallel import multiThreadCommands
from HARKestimation import minimizeNelderMead
from HealthInvEstimation import EstimationAgentType, convertVecToDict
import LoadHealthInvData as Data
import HealthInvParams as Params
import matplotlib.pyplot as plt
import pylab

# Make an extension to the estimation agent class to enable this pre-estimation exercise
class PreEstimationAgentType(EstimationAgentType):
    '''
    A very simple extension that adds or overwrites a couple methods.
    '''
    def estimationAction(self):
        '''
        Run some commands for the health production pre-estimation.
        '''
        self.update()
        self.solve()
        self.makeFOCvalues()
        self.delSolution()
        
    def altSimOnePeriod(self):
        '''
        Runs an alternate version of some of the simulation methods for the base class.
        '''
        HlvlNow = self.HlvlNow
        N = HlvlNow.size
        HealthShkStd = self.HealthShkStd0 + self.HealthShkStd1*HlvlNow
        hShkNow = np.reshape(drawNormal(N,seed=self.RNG.randint(0,2**31-1)),HlvlNow.shape)*HealthShkStd
        MedShkBase = np.reshape(drawNormal(N,seed=self.RNG.randint(0,2**31-1)),HlvlNow.shape)
        
        hLvlNow = np.maximum(np.minimum(HlvlNow + hShkNow,1.0),0.001)
        hLvlNow[HlvlNow == 0.] = np.nan
        t = self.t_sim
        MedShkMean = self.MedShkMeanFunc[t](hLvlNow)
        MedShkStd = self.MedShkStdFunc(hLvlNow)
        LogMedShkNow = MedShkMean + MedShkBase*MedShkStd
        MedShkNow = np.exp(LogMedShkNow)
        bLvlNow = self.Rfree*self.aLvlNow + self.IncomeNow[t]        
        
        PremiumNow = self.PremiumFunc[t](hLvlNow)
        CopayNow = self.CopayFunc[t](hLvlNow)
        cLvlNow, MedLvlNow, iLvlNow, xLvlNow = self.solution[t].PolicyFunc(bLvlNow,hLvlNow,MedShkNow)
        iLvlNow = np.maximum(iLvlNow,0.)
        
        MedPriceEff = self.MedPrice[t]*CopayNow
        MedShkEff = MedShkNow*MedPriceEff
        cEffNow = (1. - np.exp(-MedLvlNow/(MedShkEff)))*cLvlNow
        BelowCfloor = cEffNow < self.Cfloor
        
        xLvlNow[BelowCfloor] = self.Dfunc(self.Cfloor*np.ones_like(MedShkEff[BelowCfloor]),MedShkEff[BelowCfloor])
        iLvlNow[BelowCfloor] = 0.0
        cShareTrans = self.bFromxFunc(xLvlNow[BelowCfloor],MedShkEff[BelowCfloor])
        q = np.exp(-cShareTrans)
        cLvlNow[BelowCfloor] = xLvlNow[BelowCfloor]/(1.+q)
        MedLvlNow[BelowCfloor] = xLvlNow[BelowCfloor]*q/(1.+q)
        
        aLvlNow = bLvlNow - PremiumNow - xLvlNow - CopayNow*self.MedPrice[t]*iLvlNow
        aLvlNow = np.maximum(aLvlNow,0.0) # Fixes those who go negative due to Cfloor help
        HlvlNow = self.ExpHealthNextFunc[t](hLvlNow) + self.HealthProdFunc(iLvlNow,hLvlNow)
        self.aLvlNow = aLvlNow
        self.HlvlNow = HlvlNow
        self.CopayNow = CopayNow

        
    def makeFOCvalues(self):
        '''
        Calculates the ratio of end-of-period marginal values (times coinsurance rate)
        and stores it as attributes of self.
        '''
        DrawsPerAgent = 50
        
        RatioAdjByWealthQuint = [np.zeros(0),np.zeros(0),np.zeros(0),np.zeros(0),np.zeros(0)]
        for t in range(15):
            self.t_sim = t
            Hprev = self.HealthArraysByAge[t]
            aPrev = self.WealthArraysByAge[t]
            N = aPrev.size
            self.HlvlNow = np.tile(np.reshape(Hprev,(N,1)),(1,DrawsPerAgent))
            self.aLvlNow = np.tile(np.reshape(aPrev,(N,1)),(1,DrawsPerAgent))
            self.ActiveNow = np.ones((N,DrawsPerAgent),dtype=bool)
            self.altSimOnePeriod()
            
            H = self.HlvlNow
            a = self.aLvlNow
            dvda = self.solution[t].dvdaFunc(a,H)
            dvdH = self.solution[t].dvdHfunc(a,H)
            RatioAdj = dvda/dvdH*self.MedPrice[t]*self.CopayNow
            
            for i in range(5):
                j = i+1
                these = self.WealthQuintArraysByAge[t] == j
                RatioAdjByWealthQuint[i] = np.concatenate((RatioAdjByWealthQuint[i],RatioAdj[these,:].flatten()))
                
        self.RatioAdjByWealthQuint = RatioAdjByWealthQuint
        
        
def makeMultiTypePreEst(params):
    '''
    Create 10 instances of the estimation agent type by splitting respondents
    by sex-income quintile.  Passes the parameter dictionary to each one.
    
    Parameters
    ----------
    params : dict
        The dictionary to be used to construct each of the 10 types.
        
    Returns
    -------
    type_list : [EstimationAgentType]
        List of 10 agent types to solve and simulate.
    '''
    type_list = []
    for n in range(10):
        temp_dict = copy(params)
        temp_dict['Sex'] = n >= 5 # males are 5,6,7,8,9
        ThisType = PreEstimationAgentType(**temp_dict)
        ThisType.IncomeNow = Data.IncomeArraySmall[n,:].tolist()
        ThisType.IncomeNext = Data.IncomeArraySmall[n,1:].tolist() + [1.]
        ThisType.addToTimeVary('IncomeNow','IncomeNext')
        ThisType.makeConstantMedPrice()
        ThisType.CohortNum = np.nan
        ThisType.IncQuint = np.mod(n,5)+1
        ThisType.seed = n

        ThisType.WealthArraysByAge = Data.WealthArraysBySexIncAge[n]
        ThisType.HealthArraysByAge = Data.HealthArraysBySexIncAge[n]
        ThisType.WealthQuintArraysByAge = Data.WealthQuintArraysBySexIncAge[n]
        
        type_list.append(ThisType)
        
    return type_list
        
        
def makeMargValueRatiosByQuintile(params):
    '''
    Generate a nested list of (adjusted) marginal value ratios by income and wealth
    quintile, for use in the health production pre-estimation
    
    Parameters
    ----------
    params : dict
        The dictionary to be used to construct each of the types.
        
    Returns
    -------
    RatioAdjByIncWealthQuint : [[np.array]]
        Nested lists of arrays with adjusted marginal value ratios.
    '''
    param_dict = convertVecToDict(params)
    type_list = makeMultiTypePreEst(param_dict)
    multiThreadCommands(type_list,['estimationAction()'],num_jobs=5)
    
    RatioAdjByIncWealthQuint = [[],[],[],[],[]]
    for i in range(5):
        for j in range(5):
            RatioAdjByIncWealthQuint[i].append(np.concatenate((type_list[i].RatioAdjByWealthQuint[j],type_list[i+5].RatioAdjByWealthQuint[j])))
    
    return RatioAdjByIncWealthQuint
    

def calcHealthProdByQuintile(HealthProd0,HealthProd1,RatioAdj):
    '''
    Calculate average health produced given health production parameters and
    a nested list of (adjusted) marginal value ratios.
    '''
    HealthProdFunc = lambda i : HealthProd1*i**HealthProd0
    MargHealthProdInvFunc = lambda x : (x/(HealthProd0*HealthProd1))**(1./(HealthProd0-1.))
    
    HealthProdByQuintile = np.zeros((5,5))
    for i in range(5):
        for j in range(5):
            RatioAdj_ij = np.maximum(RatioAdj[i][j],0.0)
            HealthInv_ij = MargHealthProdInvFunc(RatioAdj_ij)
            HealthInv_ij[RatioAdj_ij == 0.] = 0.
            HealthProd_ij = HealthProdFunc(HealthInv_ij)
            HealthProdByQuintile[i,j] = np.mean(HealthProd_ij)
    
    return HealthProdByQuintile


def optimizeHealthProdParams(LifeUtility):
    '''
    Find the optimal HealthProd0 and HealthProd1 given a value of LifeUtility
    by minimizing the distance between simulated and empirical "average health
    produced".  Returns those parameters and the (upward scaled) distance.
    '''
    param_vec = Params.test_param_vec
    param_vec[3] = LifeUtility
    RatioAdjList = makeMargValueRatiosByQuintile(param_vec)
    #temp_f = lambda x : np.sum((100*(calcHealthProdByQuintile(np.exp(x[0]),np.exp(x[1]),RatioAdjList) - Data.AvgResidualByIncWealth))**2)
    def temp_f(x):
        SimMoments = calcHealthProdByQuintile(np.exp(x[0]),np.exp(x[1]),RatioAdjList)
        MomentDiff = np.reshape(SimMoments - Data.AvgResidualByIncWealth,(25,1))
        WeightedMomentSum = np.dot(MomentDiff.transpose(),np.dot(Data.W,MomentDiff))[0,0]
        return WeightedMomentSum
        
    guess = np.array([-3.5,-3.5])
    opt_params = minimizeNelderMead(temp_f,guess)
    f_opt = temp_f(opt_params)
    #print(calcHealthProdByQuintile(np.exp(opt_params[0]),np.exp(opt_params[1]),RatioAdjList))
    
    HealthProd0 = opt_params[0]
    HealthProd1 = opt_params[1]
    
    return HealthProd0, HealthProd1, f_opt


def makeContourPlot(LifeUtility):
    '''
    Make a contour plot of the pre-estimation objective function to test identification.
    '''
    param_vec = Params.test_param_vec
    param_vec[3] = LifeUtility
    RatioAdjList = makeMargValueRatiosByQuintile(param_vec)    
    #temp_f = lambda x : np.sum(((calcHealthProdByQuintile(np.exp(x[0]),np.exp(x[1]),RatioAdjList) - Data.AvgResidualByIncWealth))**2)
    def temp_f(x):
        SimMoments = calcHealthProdByQuintile(np.exp(x[0]),np.exp(x[1]),RatioAdjList)
        MomentDiff = np.reshape(SimMoments - Data.AvgResidualByIncWealth,(25,1))
        WeightedMomentSum = np.dot(MomentDiff.transpose(),np.dot(Data.W,MomentDiff))[0,0]
        return WeightedMomentSum
    
    HealthProd0_min = -3.0
    HealthProd0_max = -1.5
    HealthProd1_min = -3.8
    HealthProd1_max = -3.5
    N = 20
    HealthProd0_vec = np.linspace(HealthProd0_min,HealthProd0_max,N)
    HealthProd1_vec = np.linspace(HealthProd1_min,HealthProd1_max,N)
    HealthProd0_mesh, HealthProd1_mesh = pylab.meshgrid(HealthProd0_vec,HealthProd1_vec)
    fit_array = np.zeros([N,N]) + np.nan
    for i in range(N):
        HealthProd0 = HealthProd0_vec[i]
        for j in range(N):
            HealthProd1 = HealthProd1_vec[j]
            x = np.array([HealthProd0,HealthProd1])
            fit_array[i,j] = temp_f(x)
            #print(i,j,fit_array[i,j])
    smm_contour = pylab.contourf(HealthProd0_vec,HealthProd1_vec,fit_array.transpose()**0.1,40)
    pylab.colorbar(smm_contour)
    pylab.xlabel('HealthProd0')
    pylab.ylabel('HealthProd1')
    pylab.show()



if __name__ == '__main__':
    
    
#    param_dict = convertVecToDict(Params.test_param_vec)
#    type_list = makeMultiTypePreEst(param_dict)
#    t_start = clock()
#    type_list[4].estimationAction()
#    t_end = clock()
#    print('Processing one type took ' + str(t_end-t_start) + ' seconds.')
    
    
#    t_start = clock()
#    RatioAdj = makeMargValueRatiosByQuintile(Params.test_param_vec)
#    t_end = clock()
#    print('Making the ratio arrays took ' + str(t_end-t_start) + ' seconds.')

#    t_start = clock()
#    HealthProd0, HealthProd1, f_opt = optimizeHealthProdParams(2.14)
#    t_end = clock()
#    print('Optimizing health production parameters for one LifeUtility took ' + str(t_end-t_start) + ' seconds.')
#    print(HealthProd0, HealthProd1, f_opt)

#    makeContourPlot(3.0)

    N = 50
    LifeUtilityVec = np.linspace(2.0,2.5,num=N)
    HealthProd0Vec = np.zeros_like(LifeUtilityVec)
    HealthProd1Vec = np.zeros_like(LifeUtilityVec)
    DistanceVec = np.zeros_like(LifeUtilityVec)
    t_start = clock()
    for j in range(N):
        LifeUtility = LifeUtilityVec[j]
        HealthProd0, HealthProd1, f_opt = optimizeHealthProdParams(LifeUtility)
        HealthProd0Vec[j] = HealthProd0
        HealthProd1Vec[j] = HealthProd1
        DistanceVec[j] = f_opt
        print(LifeUtility, HealthProd0, HealthProd1, f_opt)
    t_end = clock()
    print('Optimizing health production parameters for ' + str(N) + ' LifeUtility values took ' + str(t_end-t_start) + ' seconds.')
        
    plt.plot(LifeUtilityVec,DistanceVec)
    plt.xlabel('LifeUtility')
    plt.ylabel('Moment distance')
    plt.show()
    
    plt.plot(LifeUtilityVec,HealthProd0Vec)
    plt.xlabel('LifeUtility')
    plt.ylabel('HealthProd0')
    plt.show()
    
    plt.plot(LifeUtilityVec,HealthProd1Vec)
    plt.xlabel('LifeUtility')
    plt.ylabel('HealthProd1')
    plt.show()
    