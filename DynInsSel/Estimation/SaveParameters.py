'''
This module has simple functions for displaying and saving structural parameters.
'''
import numpy as np

param_names = ['DiscFac',
               'CRRA',
               'MedCurve',
               'log(ChoiceShkMag)',
               'Cfloor',
               'log(EmpContr)',
               'UNUSED',
               'BequestShift',
               'BequestScale',
               'MedShkMean_0',
               'MedShkMean_a1',
               'MedShkMean_a2',
               'MedShkMean_a3',
               'MedShkMean_a4',
               'MedShkMean_VG0',
               'MedShkMean_VGa1',
               'MedShkMean_GD0',
               'MedShkMean_GDa1',
               'MedShkMean_FR0',
               'MedShkMean_FRa1',
               'MedShkMean_PR0',
               'MedShkMean_PRa1',
               'MedShkStd_0',
               'MedShkStd_a1',
               'MedShkStd_a2',
               'MedShkStd_a3',
               'MedShkStd_a4',
               'MedShkStd_VG0',
               'MedShkStd_VGa1',
               'MedShkStd_GD0',
               'MedShkStd_GDa1',
               'MedShkStd_FR0',
               'MedShkStd_FRa1',
               'MedShkStd_PR0',
               'MedShkStd_PRa1'
               ]
max_name_length = 18

param_tex = [  '\\beta',
               '\\rho',
               '\\nu/\\rho',
               '\\log(\\sigma^2_\\epsilon)',
               '\\underline{c}',
               '\\log(s)',
               'UNUSED',
               '\\omega_0',
               '\\omega_1',
               '\\gamma^{E}_0',
               '\\gamma^{E}_1',
               '\\gamma_{2}',
               '\\gamma_{3}',
               '\\gamma_{4}',
               '\\gamma^{V}_0',
               '\\gamma^{V}_1',
               '\\gamma^{G}_0',
               '\\gamma^{G}_1',
               '\\gamma^{F}_0',
               '\\gamma^{F}_1',
               '\\gamma^{P}_0',
               '\\gamma^{P}_1',
               '\\delta^{E}_0',
               '\\delta^{E}_1',
               '\\delta_{2}',
               '\\delta_{3}',
               '\\delta_{4}',
               '\\delta^{V}_0',
               '\\delta^{V}_1',
               '\\delta^{G}_0',
               '\\delta^{G}_1',
               '\\delta^{F}_0',
               '\\delta^{F}_1',
               '\\delta^{P}_0',
               '\\delta^{P}_1',
               ]

param_descs = ['Intertemporal discount factor',
               'Coefficient of relative risk aversion for consumption',
               'Ratio of CRRA for medical care to CRRA for consumption',
               'Log stdev of taste shocks over insurance contracts',
               'Consumption floor (10,000 USD)',
               'Log of employer contribution to ESI (10,000 USD)',
               'UNUSED',
               'Constant term in bequest motive',
               'Scale of bequest motive',
               'Excellent health constant for log mean med shock',
               'Excellent health linear coefficient on age for log mean med shock',
               'Quadratic coefficient on age for log mean medical need shock',
               'Cubic coefficient on age for log mean medical need shock',
               'Quartic coefficient on age for log mean medical need shock',
               'Very good health constant for log mean medical need shock',
               'Very good health linear age coefficient for log mean med shock',
               'Good health constant for log mean medical need shock',
               'Good health linear age coefficient for log mean med shock',
               'Fair health constant for log mean medical need shock',
               'Fair health linear age coefficient for log mean med shock',
               'Poor health constant for log mean medical need shock',
               'Poor health linear age coefficient for log mean med shock',
               'Excellent health constant for log stdev med shock',
               'Excellent health linear coefficient on age for log stdev med shock',
               'Quadratic coefficient on age for log stdev medical need shock',
               'Cubic coefficient on age for log stdev medical need shock',
               'Quartic coefficient on age for log stdev medical need shock',
               'Very good health constant for log stdev medical need shock',
               'Very good health linear age coefficient for log stdev med shock',
               'Good health constant for log stdev medical need shock',
               'Good health linear age coefficient for log stdev med shock',
               'Fair health constant for log stdev medical need shock',
               'Fair health linear age coefficient for log stdev med shock',
               'Poor health constant for log stdev medical need shock',
               'Poor health linear age coefficient for log stdev med shock',
               ]

def makeParameterString(Parameters):
    '''
    Construct a string that represents parameters as a numpy array with comments.
    '''
    out = 'test_param_vec = np.array([\n'
    for j in range(35):
        this_line = '          '
        this_line += str(Parameters[j]) + ','
        N = len(this_line)
        padding = 30 - N
        for p in range(padding):
            this_line += ' '
        this_line += ' # '
        if j < 10:
            this_line += ' '
        this_line += str(j) + ' ' + param_names[j]
        padding = max_name_length - len(param_names[j])
        for p in range(padding):
            this_line += ' '
        this_line += ': ' + param_descs[j] + '\n'
        out += this_line
    out += '])\n'
    return out


def writeParametersToFile(Parameters,filename):
    '''
    Write a vector of structural parameters to a named file.
    '''
    parameter_string = makeParameterString(Parameters)
    with open(filename,'w') as f:
        f.write(parameter_string)
        f.close()
        
        
def paramStr(value):
    '''
    Format a parameter value as a string.
    '''
    try:
        n = int(np.floor(np.log(np.abs(value))/np.log(10.)))
    except:
        n = 0   
    if n < -1:
        temp = value/10.**n
        out = "{:.2f}".format(temp) + "e" + str(n)
    else:
        out = "{:.3f}".format(value)
    return out
        
        
def makeParamTable(filename,values,stderrs=None):
    '''
    Make a txt file with tex code for the parameter table, including standard errors.
    
    Parameters
    ----------
    filename : str
        Name of file in which to store
    values : np.array
        Vector of parameter values.
    stderrs : np.array
        Vector of standard errors.
        
    Returns
    -------
    None
    '''
    if stderrs is None:
        stderrs = np.zeros_like(values) + np.nan
    
    output =  '\\begin{table} \n'
    output += '\\caption{Parameters Estimated by SMM} \n \\label{table:SMMestimates} \n'
    output += '\\centering \n'
    output += '\\small \n'
    output += '\\begin{tabular}{cccl} \n'
    output += '\\hline \\hline \n'
    output += 'Parameter & Estimate & Std Err & Description \n'
    output += '\\\\ \\hline \n'
    for j in range(len(values)):
        if np.isnan(stderrs[j]):
            se = '(---)'
        else:
            se = '(' + paramStr(stderrs[j]) + ')'
        if j > 0:
            output += '\\\\'
        output += '$' + param_tex[j] + '$ & ' + paramStr(values[j]) + ' & ' + se + ' & ' + param_descs[j] + '\n'
    output += '\\\\ \\hline \\hline \n'
    output += '\\end{tabular} \n'
    output += '\\end{table} \n'
    
    with open('./' + filename + '.txt.','w') as f:
        f.write(output)
        f.close()

    
