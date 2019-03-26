'''
This is a test of the new utility form.
'''
import sys 
sys.path.insert(0,'../../')

import numpy as np
from HARKutilities import approxLognormal
from scipy.optimize import brentq
import matplotlib.pyplot as plt

ZeroFunc = lambda c: 1. + (c*nu)/(p*Shk) - np.exp((x-c)/(p*Shk))
x = 20.
p = 1.0
Shk = 0.15
nu = 3.0
ShkDstn = approxLognormal(N=20,tail_N=10,mu=-4.0,sigma=1.5)
ShkVec = ShkDstn[1]
xVec = np.linspace(0.001,200,200)
cVec = np.zeros_like(xVec)

#for j in range(ShkVec.size):
#    Shk = ShkVec[j]
#    cVec[j] = brentq(ZeroFunc,0.0,x)
#    
#plt.plot(ShkVec,cVec)
#plt.show()
#
#plt.plot(np.log((x-cVec)/p),np.cumsum(ShkDstn[0]))
#plt.show()

for j in range(xVec.size):
    x = xVec[j]
    cVec[j] = brentq(ZeroFunc,0.0,x)

bVec = -np.log(xVec/cVec-1.)
    
plt.plot(xVec,cVec)
plt.show()

plt.plot(xVec,(xVec-cVec)/p)
plt.show()

plt.plot(xVec,bVec)
plt.show()
