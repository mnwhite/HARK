'''
This module has simple functions for displaying and saving structural parameters.
'''

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

param_descs = ['Intertemporal discount factor',
               'Coefficient of relative risk aversion for consumption',
               'Ratio of CRRA for medical care to CRRA for consumption',
               'Log stdev of taste shocks over insurance contracts',
               'Consumption floor ($10,000)',
               'Log of employer contribution to ESI ($10,000)',
               'UNUSED',
               'Constant term in bequest motive',
               'Scale of bequest motive',
               'Constant term for log mean medical need shock',
               'Linear coefficient on age for log mean medical need shock',
               'Quadratic coefficient on age for log mean medical need shock',
               'Cubic coefficient on age for log mean medical need shock',
               'Quartic coefficient on age for log mean medical need shock',
               'Very good health shifter for log mean medical need shock',
               'Very good health linear age coefficient shifter for log mean med shock',
               'Good health shifter for log mean medical need shock',
               'Good health linear age coefficient shifter for log mean med shock',
               'Fair health shifter for log mean medical need shock',
               'Fair health linear age coefficient shifter for log mean med shock',
               'Poor health shifter for log mean medical need shock',
               'Poor health linear age coefficient shifter for log mean med shock',
               'Constant term for log stdev medical need shock',
               'Linear coefficient on age for log stdev medical need shock',
               'Quadratic coefficient on age for log stdev medical need shock',
               'Cubic coefficient on age for log stdev medical need shock',
               'Quartic coefficient on age for log stdev medical need shock',
               'Very good health shifter for log stdev medical need shock',
               'Very good health linear age coefficient shifter for log stdev med shock',
               'Good health shifter for log stdev medical need shock',
               'Good health linear age coefficient shifter for log stdev med shock',
               'Fair health shifter for log stdev medical need shock',
               'Fair health linear age coefficient shifter for log stdev med shock',
               'Poor health shifter for log stdev medical need shock',
               'Poor health linear age coefficient shifter for log stdev med shock',
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
    
