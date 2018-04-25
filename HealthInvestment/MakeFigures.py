'''
This module has functions for producing figures for the Ounce of Prevention project.
'''

import numpy as np
import matplotlib.pyplot as plt

def makeSimpleFigure(data,names,x_vals,x_label,title,convert_dollars,file_name):
    '''
    Make a simple line plot with counterfactual results across several policies.
    Saves in the /Figures directory with given name, in pdf format.
    
    Parameters
    ----------
    data : [np.array]
        One or more 1D arrays with counterfactual outcomes (overall means).
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
    for n in range(N):
        if convert_dollars:
            temp_data = data[n]*10000
        else:
            temp_data = data[n]
        plt.plot(x_vals,temp_data,'-')
    plt.xlim([x_vals[0],x_vals[-1]])
    plt.xlabel(x_label, fontsize=14)
    if convert_dollars:
        plt.ylabel('USD (2000)', fontsize=14)
    else:
        plt.ylabel('Years', fontsize=14)
    plt.title(title,fontsize=14)
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
        if convert_dollars:
            temp_data = data[n]*10000
        else:
            temp_data = data[n]
        top_y = temp_data + bottom_y
        polygon_y = np.concatenate([top_y,np.flipud(bottom_y)])
        plt.fill(polygon_x,polygon_y)
        bottom_y = top_y
        
    plt.xlim([x_vals[0],x_vals[-1]])
    plt.xlabel(x_label, fontsize=14)
    if convert_dollars:
        plt.ylabel('USD (2000)', fontsize=14)
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
    var_names = ['Total medical expenses',
                 'OOP medical expenses (-)',
                 'Life expectancy',
                 'Medicare costs',
                 'Direct subsidy costs',
                 'Welfare costs',
                 'Total government costs']
    
    makeSimpleFigure([data[0],data[1],data[6]],[var_names[0],var_names[1],var_names[6]],x_vals,x_label,'Per Capita Change in PDV of Medical Expenses, ' + title_base, True, file_base + 'TotalVsOOPChange')
    makeSimpleFigure([data[2]],[var_names[2]],x_vals,x_label,'Average Change in Life Expectancy, ' + title_base, False, file_base + 'LifeExp')
    makeCumulativeFigure([data[4],data[3]],[var_names[4],var_names[3]], x_vals, x_label, 'Composition of Government Costs, ', True, 'GovtComp')
    
    
    
    