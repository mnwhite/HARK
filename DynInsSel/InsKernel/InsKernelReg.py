'''
This script runs some kernel regressions to extract insurance functions.
'''
import sys 
sys.path.insert(0,'../')

import csv
import numpy as np
from HARKutilities import kernelRegression, plotFunc, plotFuncs

infile = open('premkernelregbig.txt','r') 
my_reader = csv.reader(infile,delimiter='\t')
all_data = list(my_reader)
infile.close()
row_count = len(all_data)
year = np.zeros(row_count-1,dtype=int)
totalmed = np.zeros(row_count-1)
OOPmed = np.zeros(row_count-1)
privmed = np.zeros(row_count-1)
premium = np.zeros(row_count-1)
othermed = np.zeros(row_count-1)
for j in range(1,row_count):
    year[j-1] = int(all_data[j][0])
    totalmed[j-1] = float(all_data[j][1])
    OOPmed[j-1] = float(all_data[j][2])
    privmed[j-1] = float(all_data[j][3])
    premium[j-1] = float(all_data[j][4])
    othermed[j-1] = float(all_data[j][5])
    

these = othermed < 200.0    
#f = kernelRegression(totalmed[these],privmed[these],top=20000,h=300)
g = kernelRegression(np.log(totalmed[these]+1),privmed[these],bot=3.0,top=11.0,h=0.15)
h = lambda x : g(np.log(x+1))
#plotFunc(h,50,60000)

#y = kernelRegression(np.log(totalmed[these]+1),privmed[these]/(totalmed[these]+1),bot=3.0,top=11.0,h=0.15)
#z = lambda x : y(np.log(x+1))
#plotFunc(z,50,60000)

altmed = privmed + othermed
p = kernelRegression(np.log(totalmed+1),altmed,bot=3.0,top=11.0,h=0.15)
r = lambda x : p(np.log(x+1))
plotFuncs([h,r],50,60000)

midprem = np.median(premium)
lowprem = premium <= midprem
highprem = premium > midprem
lowf = kernelRegression(np.log(totalmed[lowprem]+1),altmed[lowprem],bot=3.0,top=11.0,h=0.15)
highf = kernelRegression(np.log(totalmed[highprem]+1),altmed[highprem],bot=3.0,top=11.0,h=0.15)
lowg = lambda x : lowf(np.log(x+1))
highg = lambda x : highf(np.log(x+1))
plotFuncs([lowg,highg],50,5000)
plotFuncs([lowg,highg],5000,40000)
