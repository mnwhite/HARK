'''
This module contains various IMI subsidy policy structures.
'''
import numpy as np
from HARKinterpolation import ConstantFunction
from copy import deepcopy

AgeCount = 95
HealthCount = 5

# Make a null subsidy policy: no subsidies for anything
NoSubsidyFunc = ConstantFunction(0.0)
NoSubsidyFunc_t = HealthCount*[NoSubsidyFunc]
NullSubsidyFuncs = AgeCount*[NoSubsidyFunc_t]

# Make a simple flat subsidy policy with constant value
FlatSubsidyFunc = ConstantFunction(0.1)
FlatSubsidyFunc_t = HealthCount*[FlatSubsidyFunc]
FlatSubsidyFuncs = AgeCount*[FlatSubsidyFunc_t]

# TODO: Make ACA-style subsidy policy

# TODO: Make BCA-style subsidy policy
