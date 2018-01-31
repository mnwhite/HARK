'''
This module has functions for making LaTeX code for various tables.
'''
import numpy as np

param_names = [
    'CRRA',
    'DiscFac',
    'MedCurve',
    'LifeUtility',
    'MargUtilityShift',
    'Cfloor',
    'Bequest0',
    'Bequest1',
    'MedShkMean0',
    'MedShkMeanSex',
    'MedShkMeanAge',
    'MedShkMeanAgeSq',
    'MedShkMeanHealth',
    'MedShkMeanHealthSq',
    'MedShkStd0',
    'MedShkStd1',
    'HealthNext0',
    'HealthNextSex',
    'HealthNextAge',
    'HealthNextAgeSq',
    'HealthNextHealth',
    'HealthNextHealthSq',
    'HealthShkStd0',
    'HealthShkStd1',
    'LogJerk',
    'LogSlope',
    'LogCurve',
    'Mortality0',
    'MortalitySex',
    'MortalityAge',
    'MortalityAgeSq',
    'MortalityHealth',
    'MortalityHealthSq'
    ]

param_tex = [
    '\\rho',
    '\\beta',
    '\\nu',
    '\\lambda',
    '\\alpha',
    '\\underline{C}',
    '\\omega_0',
    '\\omega_1',
    '\\mu_0',
    '\\mu_s',
    '\\mu_{a1}',
    '\\mu_{a2}',
    '\\mu_{h1}',
    '\\mu_{h2}',
    '\\sigma_0',
    '\\sigma_h',
    '\\gamma_0',
    '\\gamma_s',
    '\\gamma_{a1}',
    '\\gamma_{a2}',
    '\\gamma_{h1}',
    '\\gamma_{h2}',
    '\\hat{\\kappa}_0',
    '\\hat{\\kappa}_1',
    '\\hat{\\kappa}_2',
    '\\varsigma_0',
    '\\varsigma_h',
    '\\theta_0',
    '\\theta_s',
    '\\theta_{a1}',
    '\\theta_{a2}',
    '\\theta_{h1}',
    '\\theta_{h2}',
    ]

param_desc = [
    'Coefficient of relative risk aversion',
    'Intertemporal discount factor (biennial)',
    'Curvature of returns to mitigative care',
    'Utility level shifter: $u(\\lambda)=0$',
    'Marginal utility shifter for health',
    'Effective consumption floor (\$10,000)',
    'Bequest motive shifter (\$10,000)',
    'Bequest motive scaler',
    'Constant, mean of log medical need shock',
    'Sex coefficient, mean of log medical need shock',
    'Age coefficient, mean of log medical need shock',
    'Age sq coefficient, mean of log medical need shock',
    'Health coefficient, mean of log medical need shock',
    'Health sq coefficient, mean of log medical need shock',
    'Constant, stdev of log medical need shock',
    'Health coefficient, stdev of log medical need shock',
    'Constant, expected next period health',
    'Sex coefficient, expected next period health',
    'Age coefficient, expected next period health',
    'Age sq coefficient, expected next period health',
    'Health coefficient, expected next period health',
    'Health sq coefficient, expected next period health',
    'Transformed third derivative of health production at $i=0$',
    'Transformed first derivative of health production at $i=0$',
    'Transformed second derivative of health production at $i=0$'
    'Constant, stdev of log medical need shock',
    'Health coefficient, stdev of log medical need shock',
    'Constant, mortality probit',
    'Sex coefficient, mortality probit',
    'Age coefficient, mortality probit',
    'Age sq coefficient, mortality probit',
    'Health coefficient, mortality probit',
    'Health sq coefficient, mortality probit'
    ]



def paramStr(value):
    '''
    Format a parameter value as a string.
    '''
    n = int(np.floor(np.log(np.abs(value))/np.log(10.)))
    if n < -2:
        temp = value/10.**n
        out = "{:.2f}".format(temp) + "$e^" + str(n) + "$"
    else:
        out = "{:.3f}".format(value)
    return out


def makeParamTable(filename,values,which,stderrs=None):
    '''
    Make a txt file with tex code for the parameter table, including standard errors.
    
    Parameters
    ----------
    values : np.array
        Vector of parameter values.
    which : np.array
        Integer array of which parameter indices those values represent.
    stderrs : np.array
        Vector of standard errors.
        
    Returns
    -------
    None
    '''
    output =  '\\begin{center} \n'
    output += '\begin{tabular}{cccl} \n'
    output += '\\hline \\hline \n'
    output += 'Parameter & Estimate & Std Err & Description \n'
    output += '\\ \\hline \n'
    for j in range(which.size):
        i = which[j]
        if stderrs is None:
            se = '(-)'
        else:
            se = '(' + paramStr(stderrs[j]) + ')'
        output += '\\ $' + param_tex[i] + '$ & ' + paramStr(values[j]) + ' & ' + se + ' & ' + param_desc[i] + '\n'
    output += '\\ \\hline \\hline \n'
    output += '\\end{tabular} \n'
    output += '\\end{center} \n'
    
    with open('./Data/' + filename + '.txt.','w') as f:
        f.write(output)
        f.close()
        
        