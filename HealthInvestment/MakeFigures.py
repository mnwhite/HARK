'''
This module has functions for producing figures for the Ounce of Prevention project.
'''

import sys
sys.path.insert(0,'../')

import numpy as np
from scipy.stats import multivariate_normal
from statsmodels.api import OLS, WLS
import matplotlib.pyplot as plt
from HARKutilities import getPercentiles
from HARKparallel import multiThreadCommands, multiThreadCommandsFake
from HealthInvEstimation import makeMultiTypeSimple, makeMultiTypeWithCohorts, convertVecToDict
import LoadHealthInvData as Data


def makeSimpleFigure(data,names,colors,x_vals,x_label,show_legend,title,convert_dollars,file_name):
    '''
    Make a simple line plot with counterfactual results across several policies.
    Saves in the /Figures directory with given name, in pdf format.
    
    Parameters
    ----------
    data : [np.array]
        One or more 1D arrays with counterfactual outcomes (overall means).
    names : [str]
        Names of the data variables, as they should appear on the legend.
    colors : [str]
        List of colors to use for each data sequence.
    x_vals : np.array
        Corresponding x-axis values for data; should be same size as data[i].
    x_label : str
        Label as it should appear on the x axis.
    show_legend : bool
        Indicator for whether the legend should be drawn.
    title : str
        Title of figure to display at top.
    convert_dollars : bool
        Whether data is in $10,000 or years.
    file_name : str
        Name to save figure as; .pdf will be appended automatically.
        
    Returns
    -------
    None
    '''
    N = len(data)
    for n in range(N):
        plt.plot(x_vals,data[n],color=colors[n])
    plt.xlim([x_vals[0],x_vals[-1]])
    plt.xlabel(x_label, fontsize=14)
    if convert_dollars:
        plt.ylabel('$10,000 (y2000)', fontsize=14)
    else:
        plt.ylabel('Years', fontsize=14)
    plt.title(title,fontsize=14)
    if show_legend:
        plt.legend(names)
    
    plt.savefig('./Figures/' + file_name + '.pdf')
    plt.show()
    
    
def makeCumulativeFigure(data,names,x_vals,x_label,title,convert_dollars,file_name):
    '''
    Make a cumulative area plot with counterfactual results across several policies.
    Saves in the /Figures directory with given name, in pdf format.
    
    Parameters
    ----------
    data : [np.array]
        One or more 1D arrays with counterfactual outcomes (overall means).
        Area plot will be stacked bottom to top in the order provided in data.
    names : [str]
        Names of the data variables, as they should appear on the legend.
    x_vals : np.array
        Corresponding x-axis values for data; should be same size as data[i].
    x_label : str
        Label as it should appear on the x axis.
    title : str
        Title of figure to display at top.
    convert_dollars : bool
        Whether data should be multiplied by $10,000.
    file_name : str
        Name to save figure as; .pdf will be appended automatically.
        
    Returns
    -------
    None
    '''
    N = len(data)
    polygon_x = np.concatenate([x_vals,np.flipud(x_vals)])
    bottom_y = np.zeros_like(x_vals)
    for n in range(N):
        top_y = temp_data + bottom_y
        polygon_y = np.concatenate([top_y,np.flipud(bottom_y)])
        plt.fill(polygon_x,polygon_y)
        bottom_y = top_y
        
    plt.xlim([x_vals[0],x_vals[-1]])
    plt.xlabel(x_label, fontsize=14)
    if convert_dollars:
        plt.ylabel('$10,000 (2000)', fontsize=14)
    else:
        plt.ylabel('Years', fontsize=14)
    plt.title(title,fontsize=14)
    plt.legend(names,loc=2)
    plt.savefig('./Figures/' + file_name + '.pdf')
    
    
def makeCounterfactualFigures(data,x_vals,x_label,title_base,file_base):
    '''
    Make several figures based on the .
    Saves in the /Figures directory with given name, in pdf format.
    
    Parameters
    ----------
    data : [np.array]
        List of 1D arrays with counterfactual outcomes (overall means).  Order:
        TotalMed, OOPmed, ExpectedLife, Medicare, Subsidy, Welfare, Govt.
    x_vals : np.array
        Corresponding x-axis values for data; should be same size as data[i].
    x_label : str
        Label as it should appear on the x axis.
    title_base : str
        Partial title for figures to be produced.
    file_base : str
        base of name to save figure as; .pdf will be appended automatically.
        
    Returns
    -------
    None
    '''
    old_colors = ['b','g','r','c','m']
    new_colors = [u'#1f77b4', u'#ff7f0e', u'#2ca02c', u'#d62728', u'#9467bd', u'#8c564b', u'#e377c2', u'#7f7f7f', u'#bcbd22', u'#17becf']
    
    var_names = ['Total medical expenses',
                 'OOP medical expenses',
                 'Overall',
                 'Medicare costs',
                 'Direct subsidy costs',
                 'Welfare costs',
                 'Total government costs']
    
    quintile_names = ['Bottom quintile',
                      'Second quintile',
                      'Third quintile',
                      'Fourth quintile',
                      'Top quintile']
    
    # Plot changes in government spending
    makeSimpleFigure([data[3],data[4],data[5],data[6]],
                     [var_names[3],var_names[4],var_names[5],var_names[6]],
                     new_colors[0:4],
                     x_vals, x_label, True,
                     'Per Capita Change in PDV of Government Spending',
                     True, file_base + 'GovtChange')
    
    # Plot changes in medical spending: total vs govt vs OOP
    makeSimpleFigure([data[0],data[1],data[6]],
                     [var_names[0],var_names[1],var_names[6]],
                     [new_colors[4],new_colors[5],new_colors[3]],
                     x_vals, x_label, True,
                     'Per Capita Change in PDV of Medical Costs',
                     True, file_base + 'MedChange')
    
    # Plot changes in life expectancy
    makeSimpleFigure([data[2]] + [data[8][:,i] for i in range(5)],
                     [var_names[2]] + quintile_names,
                     ['k'] + old_colors,
                     x_vals, x_label, True,
                     'Average Change in Life Expectancy by Income',
                     False, file_base + 'LifeExp')
    
    # Plot changes in life expectancy
    makeSimpleFigure([data[2]] + [data[9][:,i] for i in range(5)],
                     [var_names[2]] + quintile_names,
                     ['k'] + old_colors,
                     x_vals, x_label, False,
                     'Average Willingness to Pay for Policy by Income',
                     True, file_base + 'WTP')
    
    #makeCumulativeFigure([data[4],data[3]],[var_names[4],var_names[3]], x_vals, x_label, 'Composition of Government Costs, ' + title_base, True, 'GovtComp')
    
    
    
def makeValidationFigures(params,use_cohorts):
    '''
    Make several figures that compare simulated outcomes from the estimated model
    to their data counterparts, for external validation.
    
    Parameters
    ----------
    params : np.array
        Size 33 array of model parameters, like that used for estimation.
    use_cohorts : bool
        Indicator for whether or not to model differences across cohorts.
        
    Returns
    -------
    None
    '''
    # Make, solve, and simulate the types
    param_dict = convertVecToDict(params)
    if use_cohorts:
        type_list = makeMultiTypeWithCohorts(param_dict)
    else:
        type_list = makeMultiTypeSimple(param_dict)
    for this_type in type_list:
        this_type.track_vars.append('MedLvlNow')
    multiThreadCommands(type_list,['estimationAction()'],num_jobs=5)
    
    # Combine simulated data across all types
    aLvlHist = np.concatenate([this_type.aLvlNow_hist for this_type in type_list],axis=1)
    hLvlHist = np.concatenate([this_type.hLvlNow_hist for this_type in type_list],axis=1)
    OOPhist  = np.concatenate([this_type.OOPmedNow_hist for this_type in type_list],axis=1)
    MortHist = np.concatenate([this_type.DiePrbNow_hist for this_type in type_list],axis=1)
    WeightHist = np.concatenate([this_type.CumLivPrb_hist for this_type in type_list],axis=1)
    MedHist  = np.concatenate([this_type.MedLvlNow_hist for this_type in type_list],axis=1)
    
    # Combine data labels across types
    HealthTert = np.concatenate([this_type.HealthTert for this_type in type_list])
    HealthQuint = np.concatenate([this_type.HealthQuint for this_type in type_list])
    WealthQuint = np.concatenate([this_type.WealthQuint for this_type in type_list])
    IncQuint = np.concatenate([this_type.IncQuintLong for this_type in type_list])
    Sex = np.concatenate([this_type.SexLong for this_type in type_list])
    
    # Combine in-data-span masking array across all types
    Active = hLvlHist > 0.
    InDataSpan = np.concatenate([this_type.InDataSpanArray for this_type in type_list],axis=1)
    WeightAdj = InDataSpan*WeightHist
    
    # Extract deciles of health by age from the simulated data
    pctiles = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
    SimHealthPctiles = np.zeros((15,len(pctiles)))
    for t in range(15):
        SimHealthPctiles[t,:] = getPercentiles(hLvlHist[t,:],weights=WeightAdj[t,:],percentiles=pctiles)
        
    # Extract deciles of health from the HRS data
    DataHealthPctiles = np.zeros((15,len(pctiles)))
    for t in range(15):
        these = np.logical_and(Data.AgeBoolArray[:,:,t], Data.Alive)
        h_temp = Data.h_data[these]
        DataHealthPctiles[t,:] = getPercentiles(h_temp,percentiles=pctiles)
        
    # Plot deciles of health by by age
    AgeVec = np.linspace(67.,95.,num=15)
    plt.plot(AgeVec,SimHealthPctiles,'-k')
    plt.plot(AgeVec,DataHealthPctiles,'--k')
    plt.ylim(0.,1.)
    plt.ylabel('Health capital $h_{it}$')
    plt.xlabel('Age')
    plt.title('Simulated vs Actual Distribution of Health by Age')
    plt.savefig('./Figures/HealthDistribution.pdf')
    plt.show()
    
    OOPmodFunc = lambda x : np.log(10000*x)
    
    # Extract many percentiles of OOP spending from the simulated data
    OOP_sim = OOPhist.flatten()
    Weight_temp = WeightAdj.flatten()
    CDFvalsSim = np.linspace(0.0001,0.999,1000)
    OOPsimCDF_A0 = getPercentiles(OOP_sim*10000,weights=Weight_temp,percentiles=CDFvalsSim)
    OOPsimCDF_B0 = getPercentiles(OOPmodFunc(OOP_sim),weights=Weight_temp,percentiles=CDFvalsSim)
    
    # Extract some percentiles of OOP spending from the HRS data
    these = np.logical_and(Data.Alive, np.logical_not(np.isnan(Data.m_data)))
    OOP_data = Data.m_data[these]
    CDFvalsData = np.linspace(0.0001,0.999,500)
    OOPdataCDF_A0 = getPercentiles(OOP_data*10000,weights=None,percentiles=CDFvalsData)
    OOPdataCDF_B0 = getPercentiles(OOPmodFunc(OOP_data),weights=None,percentiles=CDFvalsData)
    
    # Plot the CDF of log out-of-pocket medical spending
    plt.subplot(211)
    plt.title('CDF of OOP Medical Spending')
    plt.plot(OOPdataCDF_B0,CDFvalsData,'-r')
    plt.plot(OOPsimCDF_B0,CDFvalsSim,'-b')
    plt.xlim(8.,11.5)
    plt.ylim(0.85,1.0)
    plt.xticks([np.log(3000), np.log(6000), np.log(12000), np.log(24000), np.log(48000), np.log(96000)],['3000','6000','12000','24000','48000','96000'])
    
    # Plot the CDF of out-of-pocket medical spending
    plt.subplot(212)
    plt.plot(OOPdataCDF_A0,CDFvalsData,'-r')
    plt.plot(OOPsimCDF_A0,CDFvalsSim,'-b')
    plt.xlim(0.,3000.)
    plt.ylim(0.0,0.9)
    plt.xlabel('Out-of-pocket medical expenses, biannual')
    plt.ylabel('Cumulative distribution')
    plt.legend(['HRS data','Model'],loc=4)
    plt.savefig('./Figures/OOPdistribution.pdf')
    plt.show()
    
    # Calculate the serial correlation of log OOP medical spending in simulated data
    Med_sim = np.log(10000*OOPhist + 1.)
    serial_corr_sim = np.zeros(15)
    serial_corr_sim_inc = np.zeros((15,5))
    for t in range(15):
        these = np.logical_and(WeightAdj[t+1,:] > 0., WeightAdj[t+1,:] < 1.) # Alive but not the first simulated period
        Med_t = Med_sim[t+1,these]
        Med_tm1 = Med_sim[t,these]
        weight_reg = WeightAdj[t+1,these]
        const_reg = np.ones_like(Med_t)
        regressors = np.transpose(np.vstack([const_reg,Med_tm1]))
        temp_model = WLS(Med_t,regressors,weights=weight_reg)
        temp_results = temp_model.fit()
        serial_corr_sim[t] = temp_results.rsquared
        for i in range(5):
            those = np.logical_and(these,IncQuint==i+1)
            Med_t = Med_sim[t+1,those]
            Med_tm1 = Med_sim[t,those]
            weight_reg = WeightAdj[t+1,those]
            const_reg = np.ones_like(Med_t)
            regressors = np.transpose(np.vstack([const_reg,Med_tm1]))
            temp_model = WLS(Med_t,regressors,weights=weight_reg)
            temp_results = temp_model.fit()
            serial_corr_sim_inc[t,i] = temp_results.rsquared
        
    # Calculate the serial correlation of log OOP medical spending in HRS data
    DataExists = np.logical_and(np.logical_not(np.isnan(Data.m_data[:-1,:])),np.logical_not(np.isnan(Data.m_data[1:,:])))
    BothAlive  = np.logical_and(Data.Alive[:-1,:],Data.Alive[1:,:])
    Usable = np.logical_and(DataExists,BothAlive)
    serial_corr_data = np.zeros(15)
    serial_corr_data_inc = np.zeros((15,5))
    Med_data = np.log(10000*Data.m_data + 1.)
    for t in range(15):
        these = np.logical_and(Usable,Data.AgeBoolArray[:-1,:,t])
        Med_t = Med_data[1:,:][these]
        Med_tm1 = Med_data[:-1,:][these]
        const_reg = np.ones_like(Med_t)
        regressors = np.transpose(np.vstack([const_reg,Med_tm1]))
        temp_model = OLS(Med_t,regressors)
        temp_results = temp_model.fit()
        serial_corr_data[t] = temp_results.rsquared
        for i in range(5):
            those = np.logical_and(these,Data.IncQuintBoolArray[:-1,:,i])
            Med_t = Med_data[1:,:][those]
            Med_tm1 = Med_data[:-1,:][those]
            const_reg = np.ones_like(Med_t)
            regressors = np.transpose(np.vstack([const_reg,Med_tm1]))
            temp_model = OLS(Med_t,regressors)
            temp_results = temp_model.fit()
            serial_corr_data_inc[t,i] = temp_results.rsquared
            
    # Make a plot of serial correlation of OOP medical expenses
    plt.subplot(3,2,1)
    plt.plot(AgeVec,serial_corr_data,'-r')
    plt.plot(AgeVec,serial_corr_sim,'-b')
    plt.ylim(0,0.5)
    plt.xticks([])
    plt.text(75,0.4,'All individuals')
    
    plt.subplot(3,2,2)
    plt.plot(AgeVec,serial_corr_data_inc[:,0],'-r')
    plt.plot(AgeVec,serial_corr_sim_inc[:,0],'-b')
    plt.ylim(0,0.5)
    plt.xticks([])
    plt.yticks([])
    plt.text(70,0.4,'Bottom income quintile')
    
    plt.subplot(3,2,3)
    plt.plot(AgeVec,serial_corr_data_inc[:,1],'-r')
    plt.plot(AgeVec,serial_corr_sim_inc[:,1],'-b')
    plt.ylim(0,0.5)
    plt.xticks([])
    plt.text(67,0.4,'Second income quintile')
    plt.ylabel('$R^2$ of regression of $\log(OOP_{t})$ on $\log(OOP_{t-1})$')
    
    plt.subplot(3,2,4)
    plt.plot(AgeVec,serial_corr_data_inc[:,2],'-r')
    plt.plot(AgeVec,serial_corr_sim_inc[:,2],'-b')
    plt.ylim(0,0.5)
    plt.xticks([])
    plt.yticks([])
    plt.text(70,0.4,'Third income quintile')
    
    plt.subplot(3,2,5)
    plt.plot(AgeVec,serial_corr_data_inc[:,3],'-r')
    plt.plot(AgeVec,serial_corr_sim_inc[:,3],'-b')
    plt.ylim(0,0.5)
    plt.xlabel('Age')
    plt.text(70,0.4,'Fourth income quintile')
    
    plt.subplot(3,2,6)
    plt.plot(AgeVec,serial_corr_data_inc[:,4],'-r')
    plt.plot(AgeVec,serial_corr_sim_inc[:,4],'-b')
    plt.ylim(0,0.5)
    plt.xlabel('Age')
    plt.yticks([])
    plt.text(70,0.4,'Top income quintile')
    plt.savefig('./Figures/SerialCorrOOP.pdf')
    plt.show()
    
    
    # Make a plot of serial correlation of OOP medical expenses
    plt.plot(AgeVec+2,serial_corr_data,'-r')
    plt.plot(AgeVec+2,serial_corr_sim,'-b')
    plt.xlabel('Age')
    plt.ylabel('$R^2$ of regression of $\log(OOP_{t})$ on $\log(OOP_{t-1})$')
    plt.legend(['HRS data','Model'],loc=1)
    plt.show()
    
    # Calculate mortality probability by age and income quintile in simulated data
    MortByIncAge_data = Data.MortByIncAge
    MortByIncAge_sim = np.zeros((5,15))
    MortByAge_sim = np.zeros(15)
    for t in range(15):
        THESE = np.logical_and(Active[t,:],InDataSpan[t,:])
        Weight = WeightHist[t+1,THESE]
        WeightSum = np.sum(Weight)
        Mort = MortHist[t+1,THESE]
        MortByAge_sim[t] = np.dot(Mort,Weight)/WeightSum
        for i in range(5):
            right_inc = IncQuint == i+1
            these = np.logical_and(THESE,right_inc)
            Mort = MortHist[t+1,these]
            Weight = WeightHist[t+1,these]
            WeightSum = np.sum(Weight)
            MortByIncAge_sim[i,t] = np.dot(Mort,Weight)/WeightSum
    
    # Plot mortality probability by age and income quintile
    income_colors = ['b','r','g','m','c']
    for i in range(5):
        plt.plot(AgeVec,MortByIncAge_sim[i,:]-MortByAge_sim,'-' + income_colors[i])
    for i in range(5):
        plt.plot(AgeVec,MortByIncAge_data[i,:]-MortByAge_sim,'.' + income_colors[i])
    plt.xlabel('Age')
    plt.ylabel('Relative death probability (biannual)')
    plt.title('Death Probability by Income Quintile')
    plt.legend(['Bottom quintile','Second quintile','Third quintile','Fourth quintile','Top quintile'],loc=2)
    plt.savefig('./Figures/MortByIncAge.pdf')
    plt.show()
    
    # Plot the 99% confidence band of the health production function
    mean  = np.array([-2.13369276099, 1.71842956397])
    covar = array([[0.02248322, 0.01628292],[0.01628308, 0.01564192]])
    dstn = multivariate_normal(mean,covar)
    


if __name__ == '__main__':
    import HealthInvParams as Params
    
    mean  = np.array([-2.13369276099, 1.71842956397])
    covar = np.array([[0.02248322, 0.01628292],[0.01628308, 0.01564192]])
    dstn  = multivariate_normal(mean,covar)
    N = 10000
    M = 201
    draws = dstn.rvs(10000)
    MedVec = np.linspace(0.,1.5,M)
    func_data = np.zeros((N,M))
    
    def makeHealthProdFunc(LogSlope,LogCurve):
        LogJerk = 15.6
        tempw = np.exp(LogJerk)
        HealthProd0 = 1. - tempw
        tempx = np.exp(LogSlope) # Slope of health production function at iLvl=0
        HealthProd2 = np.exp(LogJerk - LogCurve)
        HealthProdFunc = lambda i : tempx/HealthProd0*((i*HealthProd2**((1.-HealthProd0)/HealthProd0) + HealthProd2**(1./HealthProd0))**HealthProd0 - HealthProd2)
        return HealthProdFunc
        
    for n in range(N):
        f = makeHealthProdFunc(draws[n,0],draws[n,1])
        func_data[n,:] = f(MedVec)
        
    f = makeHealthProdFunc(Params.test_param_vec[25],Params.test_param_vec[26])
    CI_array = np.zeros((M,2))
    for m in range(M):
        CI_array[m,:] = getPercentiles(func_data[:,m],percentiles=[0.025,0.975])
    health_prod = f(MedVec)
    
    plt.plot(MedVec,health_prod,'-r')
    plt.plot(MedVec,CI_array[:,0],'--k',linewidth=0.5)
    plt.plot(MedVec,CI_array[:,1],'--k',linewidth=0.5)
    plt.xlim([-0.005,1.5])
    plt.ylim([0.,None])
    plt.xlabel('Health investment $n_{it}$, \$10,000 (y2000)')
    plt.ylabel('Health produced ')
    plt.title('Estimated Health Production Function')
    plt.legend(['Estimated health production function','Pointwise 95% confidence bounds'],loc=4)
    plt.savefig('./Figures/HealthProdFunc.pdf')
    plt.show()
    
    
       

    #makeValidationFigures(Params.test_param_vec,False)    