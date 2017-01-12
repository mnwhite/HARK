'''
This script creates Stata do files to process the MEPS individual and PRPL data.
'''

import csv

infile = open('VarListInd.txt','r') 
my_reader = csv.reader(infile,delimiter='\t')
all_data = list(my_reader)
infile.close()
row_count = len(all_data)
col_count = len(all_data[0])
var_name_list = []
for j in range(1,row_count):
    var_name_list.append(all_data[j][1])
    
infile = open('VarListPlan.txt','r') 
my_reader = csv.reader(infile,delimiter='\t')
plan_data = list(my_reader)
infile.close()
row_count_a = len(plan_data)
col_count_a = len(plan_data[0])

for y in range(2,col_count):
    year = all_data[0][y]
    yearnum = int(year)
    panel1 = yearnum - 1996
    panel2 = yearnum - 1995
    this_year_vars = []
    plan_vars = []
    for j in range(1,row_count):
        this_year_vars.append(all_data[j][y])
    for j in range(1,row_count_a):
        plan_vars.append(plan_data[j][y-1])
        
    script = 'clear\nset more off\n'
    script += 'use plans' + year + '.dta\nkeep '
    for j in range(1,row_count_a):
        script += plan_data[j][y-1] + ' '
    script += '\n\n'
    script += 'keep if OOPPREMX >= 0\nkeep if PHOLDER == 1\nbysort DUPERSID: egen premR3 = sum(OOPPREMX) if RN == 3\nbysort DUPERSID: egen premR1 = sum(OOPPREMX) if RN == 1\ngen totalprem = 0\n'
    script += 'replace totalprem = premR3 if PANEL == ' + str(panel1) + '\n'
    script += 'replace totalprem = premR1 if PANEL == ' + str(panel2) + '\n'
    script += 'by DUPERSID: keep if _n == 1\nkeep DUPERSID totalprem\nsort DUPERSID\n'
    script += 'save tempprem' + year + '.dta, replace\n\n'
    
    script += 'use raw' + year + '.dta\nkeep '
    for j in range(len(this_year_vars)):
        script += this_year_vars[j] + ' '
    script += '\n\n'
    for j in range(len(this_year_vars)):
        script += 'rename ' + this_year_vars[j] + ' ' + var_name_list[j] + '\n'
    script += 'gen year = ' + year + '\norder year\n'
    script += '\nsort DUPERSID\nmerge 1:1 DUPERSID using tempprem' + year + '.dta\n'
    script += 'save ind' + year + '.dta, replace\n'
    
    filename = 'processind' + year + '.do'
    f = open(filename,'w')
    f.write(script)
    f.close()

    