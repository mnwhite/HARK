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
    '\\varsigma',
    '\\alpha',
    '\\underline{c}',
    '\\omega_0',
    '\\omega_1',
    '\\gamma_0',
    '\\gamma_s',
    '\\gamma_{j1}',
    '\\gamma_{j2}',
    '\\gamma_{h1}',
    '\\gamma_{h2}',
    '\\gamma_{\\sigma 0}',
    '\\gamma_{\\sigma 1}',
    '\\delta_0',
    '\\delta_s',
    '\\delta_{j1}',
    '\\delta_{j2}',
    '\\delta_{h1}',
    '\\delta_{h2}',
    '\\delta_{\\sigma_0}',
    '\\delta_{\\sigma_1}',
    '\\lambda_3',
    '\\lambda_1',
    '\\lambda_2',
    '\\theta_0',
    '\\theta_s',
    '\\theta_{j1}',
    '\\theta_{j2}',
    '\\theta_{h1}',
    '\\theta_{h2}',
    ]

param_desc = [
    'CRRA for consumption $c$',
    'Intertemporal discount factor (biennial)',
    'CRRA for medical consumption $m$',
    'Utility level shifter: $U(\\varsigma,m;0)=0$',
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
    'Constant, stdev of health shock',
    'Health coefficient, stdev of health shock',
    'Transformed exponent of health production function',
    'Log slope of health production function at $n=0$',
    'Log curvature of health production function at $n=0$',
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
        if j == 4:
            continue
        if np.isnan(stderrs[j]):
            se = '(---)'
        else:
            se = '(' + paramStr(stderrs[j]) + ')'
        if j > 0:
            output += '\\\\'
        output += '$' + param_tex[j] + '$ & ' + paramStr(values[j]) + ' & ' + se + ' & ' + param_desc[j] + '\n'
    output += '\\\\ \\hline \\hline \n'
    output += '\\end{tabular} \n'
    output += '\\end{table} \n'
    
    with open('./Tables/' + filename + '.txt.','w') as f:
        f.write(output)
        f.close()
        
        
def makeCounterfactualSummaryTablesOneVar(means,var_name,spec_name,file_name,label,convert_dollars=True):
    '''
    Make two tables showing decomposed means of a simulation outcome variable:
    income-wealth and income-health.  Saves to txt files in the /Tables directory.
    
    Parameters
    ----------
    means : MyMeans
        Object containing overall and decomposed outcomes.
    var_name : str
        Name of the variable of interest as it should appear in the table.
    spec_name : str
        Name of the counterfactual policy as it should appear in the table.
    file_name : str
        Name of the file to be saved; .txt extension will be added automatically.
    label : str
        LaTeX label for the table.
    convert_dollars : bool
        Whether to convert values in means to $10k of dollars.
        
    Returns
    -------
    None
    '''
    if convert_dollars:
        f = lambda x : '\$' + str(int(np.round(x*10000)))
    else:
        f = lambda x : "{:.2f}".format(x)
    
    # Make the income-health table
    IH = means.byIncHealth
    I = means.byIncome
    H = means.byHealth
    O = means.overall
    table1 = '\\begin{table} \n \\caption{' + var_name + ' by Income and Health, ' + spec_name + '}\n \\label{table:' + label + 'IH}\n'
    table1 += '\\centering \n'
    table1 += '\\begin{tabular}{l c c c c c} \n'
    table1 += '\\hline \\hline \n'
    table1 += 'Income & \multicolumn{5}{c}{Range of Health $h$} \\\\ \n'
    table1 += 'Quintile & All & $(0,0.25]$ & $(0.25,0.5]$ & $(0.5,0.75]$ & $(0.75,1.0]$ \\\\ \n'
    table1 += '\\hline \n'
    table1 += 'Bottom & ' + f(I[0]) + ' & ' + f(IH[0][0]) + ' & ' + f(IH[0][1]) + ' & ' + f(IH[0][2]) + ' & ' + f(IH[0][3]) + ' \\\\ \n'
    table1 += 'Second & ' + f(I[1]) + ' & ' + f(IH[1][0]) + ' & ' + f(IH[1][1]) + ' & ' + f(IH[1][2]) + ' & ' + f(IH[1][3]) + ' \\\\ \n'
    table1 += 'Third  & ' + f(I[2]) + ' & ' + f(IH[2][0]) + ' & ' + f(IH[2][1]) + ' & ' + f(IH[2][2]) + ' & ' + f(IH[2][3]) + ' \\\\ \n'
    table1 += 'Fourth & ' + f(I[3]) + ' & ' + f(IH[3][0]) + ' & ' + f(IH[3][1]) + ' & ' + f(IH[3][2]) + ' & ' + f(IH[3][3]) + ' \\\\ \n'
    table1 += 'Top    & ' + f(I[4]) + ' & ' + f(IH[4][0]) + ' & ' + f(IH[4][1]) + ' & ' + f(IH[4][2]) + ' & ' + f(IH[4][3]) + ' \\\\ \n'
    table1 += '\\hline \n'
    table1 += 'All    & ' + f(O) + ' & ' + f(H[0]) + ' & ' + f(H[1]) + ' & ' + f(H[2]) + ' & ' + f(H[3]) + ' \\\\ \n'
    table1 += '\\hline \\hline \n'
    table1 += '\\end{tabular} \n'
    table1 += '\\end{table} \n'
    g = open('./Tables/' + file_name + 'IncHealth.txt','w')
    g.write(table1)
    g.close()
    
    # Make the income-wealth table
    IW = means.byIncWealth
    table2 =  '\\begin{table} \n \\caption{' + var_name + ' by Income and Wealth, ' + spec_name + '}\n \\label{table:' + label + 'IW} \n'
    table2 += '\\centering \n'
    table2 += '\\begin{tabular}{l c c c c c} \n'
    table2 += '\\hline \\hline \n'
    table2 += 'Income & \multicolumn{5}{c}{Wealth Quintile} \\\\ \n'
    table2 += 'Quintile & Bottom & Second & Third & Fourth & Top \\\\ \n'
    table2 += '\\hline \n'
    table2 += 'Bottom & ' + f(IW[0][0]) + ' & ' + f(IW[0][1]) + ' & ' + f(IW[0][2]) + ' & ' + f(IW[0][3]) + ' & ' + f(IW[0][4]) + ' \\\\ \n'
    table2 += 'Second & ' + f(IW[1][0]) + ' & ' + f(IW[1][1]) + ' & ' + f(IW[1][2]) + ' & ' + f(IW[1][3]) + ' & ' + f(IW[1][4]) + ' \\\\ \n'
    table2 += 'Third  & ' + f(IW[2][0]) + ' & ' + f(IW[2][1]) + ' & ' + f(IW[2][2]) + ' & ' + f(IW[2][3]) + ' & ' + f(IW[2][4]) + ' \\\\ \n'
    table2 += 'Fourth & ' + f(IW[3][0]) + ' & ' + f(IW[3][1]) + ' & ' + f(IW[3][2]) + ' & ' + f(IW[3][3]) + ' & ' + f(IW[3][4]) + ' \\\\ \n'
    table2 += 'Top    & ' + f(IW[4][0]) + ' & ' + f(IW[4][1]) + ' & ' + f(IW[4][2]) + ' & ' + f(IW[4][3]) + ' & ' + f(IW[4][4]) + ' \\\\ \n'
    table2 += '\\hline \\hline \n'
    table2 += '\\end{tabular} \n'
    table2 += '\\end{table} \n'
    g = open('./Tables/' + file_name + 'IncWealth.txt','w')
    g.write(table2)
    g.close()
    
    
def makeTableBySexIncHealth(means,var_name,file_name,label,convert_dollars=True):
    '''
    Make one table showing decomposed means of a simulation outcome variable:
    sex-income-health.  Saves to txt files in the /Tables directory.
    
    Parameters
    ----------
    means : MyMeans
        Object containing overall and decomposed outcomes.
    var_name : str
        Name of the variable of interest as it should appear in the table.
    spec_name : str
        Name of the counterfactual policy as it should appear in the table.
    file_name : str
        Name of the file to be saved; .txt extension will be added automatically.
    label : str
        LaTeX label for the table.
    convert_dollars : bool
        Whether to convert values in means to $10k of dollars.
        
    Returns
    -------
    None
    '''
    f = lambda x : "{:.1f}".format(x)
    
    # Make the sex-income-health table
    SIH = means.bySexIncHealth
    I = means.byIncome
    SH = means.bySexHealth
    O = means.overall
    
    table1 = '\\begin{table} \n \\caption{' + var_name + ' by Sex, Income, and Health}\n \\label{table:' + label + 'SIH}\n'
    table1 += '\\centering \n'
    table1 += '\\begin{tabular}{l c c c c c} \n'
    table1 += '\\hline \\hline \n'
    table1 += 'Income & \multicolumn{2}{c}{Women} & \multicolumn{2}{c}{Men} & \\\\ \n'
    table1 += 'Quintile & $h < 0.5$ & $h \\geq 0.5$ & $h < 0.5$ & $h \\geq 0.5$ & All \\\\ \n'
    table1 += '\\hline \n'
    table1 += 'Bottom & ' + f(SIH[0,0,0]) + ' & ' + f(SIH[0,0,1]) + ' & ' + f(SIH[1,0,0]) + ' & ' + f(SIH[1,0,1]) + ' & ' + f(I[0]) + ' \\\\ \n'
    table1 += 'Second & ' + f(SIH[0,1,0]) + ' & ' + f(SIH[0,1,1]) + ' & ' + f(SIH[1,1,0]) + ' & ' + f(SIH[1,1,1]) + ' & ' + f(I[1]) + ' \\\\ \n'
    table1 += 'Third  & ' + f(SIH[0,2,0]) + ' & ' + f(SIH[0,2,1]) + ' & ' + f(SIH[1,2,0]) + ' & ' + f(SIH[1,2,1]) + ' & ' + f(I[2]) + ' \\\\ \n'
    table1 += 'Fourth & ' + f(SIH[0,3,0]) + ' & ' + f(SIH[0,3,1]) + ' & ' + f(SIH[1,3,0]) + ' & ' + f(SIH[1,3,1]) + ' & ' + f(I[3]) + ' \\\\ \n'
    table1 += 'Top    & ' + f(SIH[0,4,0]) + ' & ' + f(SIH[0,4,1]) + ' & ' + f(SIH[1,4,0]) + ' & ' + f(SIH[1,4,1]) + ' & ' + f(I[4]) + ' \\\\ \n'
    table1 += '\\hline \n'
    table1 += 'All    & ' + f(SH[0,0]) + ' & ' + f(SH[0,1]) + ' & ' + f(SH[1,0]) + ' & ' + f(SH[1,1]) + ' & ' + f(O) + ' \\\\ \n'
    table1 += '\\hline \\hline \n'
    table1 += '\\end{tabular} \n'
    table1 += '\\end{table} \n'
    g = open('./Tables/' + file_name + 'SexIncHealth.txt','w')
    g.write(table1)
    g.close()
    
    
        
def makeCounterfactualSummaryTables(means,spec_name,file_base,label):
    '''
    Make two tables showing decomposed means of a simulation outcome variable:
    income-wealth and income-health.  Saves to txt files in the /Tables directory.
    
    Parameters
    ----------
    means : [MyMeans]
        Objects containing overall and decomposed outcomes.  Order: TotalMed, OOPmed,
        ExpectedLife, Medicare, Subsidy, Welfare, Govt.
    spec_name : str
        Name of the counterfactual policy as it should appear in the table.
    file_base : str
        Base name of the file to be saved; .txt extension will be added automatically.
    label : str
        LaTeX label base for the table.
        
    Returns
    -------
    None
    '''
    var_names = ['Change in PDV of Total Medical Expenses',
                 'Change in PDV of Out of Pocket Medical Expenses',
                 'Change in Remaining Life Expectancy (Years)',
                 'Change in PDV of Medicare Costs',
                 'PDV of Direct Subsidy Expenses',
                 'Change in PDV of Welfare Payments',
                 'Change in PDV of Total Government Expenses',
                 'Willingness to Pay for Policy',
                 'Remaining Life Expectancy']
    var_codes = ['TotalMed',
                 'OOPmed',
                 'ExpLife',
                 'Medicare',
                 'Subsidy',
                 'Welfare',
                 'Govt',
                 'WTP',
                 'LifeBase']
    convert = [True,
               True,
               False,
               True,
               True,
               True,
               True,
               True,
               False]
    
    for i in range(8):
        makeCounterfactualSummaryTablesOneVar(means[i],var_names[i],spec_name, file_base + var_codes[i], label + var_codes[i],convert[i])
    makeTableBySexIncHealth(means[8], var_names[8], var_codes[8], label + var_codes[8], convert[8])
          