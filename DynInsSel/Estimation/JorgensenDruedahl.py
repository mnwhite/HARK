'''
This module calls OpenCL code to run the Jorgensen-Druedahl convexity fix for
the DynInsSel model.
'''

import sys
import os
import numpy as np
import opencl4py as cl
import matplotlib.pyplot as plt
#from copy import copy
os.environ["PYOPENCL_CTX"] = "0:2" # This is where you choose a device number
sys.path.insert(0, os.path.abspath('../'))
sys.path.insert(0,'../../')
sys.path.insert(0, os.path.abspath('./'))
from HARKinterpolation import BilinearInterp

f = open('JorgensenDruedahl.cl')
program_code = f.read()
f.close()

# Make a context and kernel
platforms = cl.Platforms()
ctx = platforms.create_some_context()
queue = ctx.create_queue(ctx.devices[0])
program = ctx.create_program(program_code)
JDkernel = program.get_kernel('doJorgensenDruedahlFix')

class JDfixer(object):
    '''
    A class-oriented implementation of the Jorgensen-Druedahl convexity fix for
    the DynInsSel model.  Each period, the solver creates an instance of this
    class, which is called whenever a JD fix is needed, passing current data
    to the buffers.
    '''
    def __init__(self,mLvlDataDim,MedShkDataDim,mGridDenseSize,ShkGridDenseSize):
        '''
        Make a new JDfixer object
        '''
        self.mLvlDataDim = mLvlDataDim
        self.MedShkDataDim = MedShkDataDim
        self.mGridDenseSize = mGridDenseSize
        self.ShkGridDenseSize = ShkGridDenseSize
        self.ThreadCount = mGridDenseSize*ShkGridDenseSize
        IntegerInputs = np.array([mLvlDataDim,MedShkDataDim,mGridDenseSize,ShkGridDenseSize,self.ThreadCount],dtype=np.int32)
        data_temp = np.zeros(mLvlDataDim*MedShkDataDim)
        out_temp = np.zeros(self.ThreadCount)
        
        # Make buffers
        self.mLvlData_buf = ctx.create_buffer(cl.CL_MEM_READ_WRITE | cl.CL_MEM_COPY_HOST_PTR,data_temp)
        self.MedShkData_buf = ctx.create_buffer(cl.CL_MEM_READ_WRITE | cl.CL_MEM_COPY_HOST_PTR,data_temp)
        self.ValueData_buf = ctx.create_buffer(cl.CL_MEM_READ_WRITE | cl.CL_MEM_COPY_HOST_PTR,data_temp)
        self.xLvlData_buf = ctx.create_buffer(cl.CL_MEM_READ_WRITE | cl.CL_MEM_COPY_HOST_PTR,data_temp)
        self.mGridDense_buf = ctx.create_buffer(cl.CL_MEM_READ_WRITE | cl.CL_MEM_COPY_HOST_PTR,np.zeros(mGridDenseSize))
        self.ShkGridDense_buf = ctx.create_buffer(cl.CL_MEM_READ_WRITE | cl.CL_MEM_COPY_HOST_PTR,np.zeros(ShkGridDenseSize))
        self.xLvlOut_buf = ctx.create_buffer(cl.CL_MEM_READ_WRITE | cl.CL_MEM_COPY_HOST_PTR,out_temp)
        self.ValueOut_buf = ctx.create_buffer(cl.CL_MEM_READ_WRITE | cl.CL_MEM_COPY_HOST_PTR,out_temp)
        self.IntegerInputs_buf = ctx.create_buffer(cl.CL_MEM_READ_ONLY | cl.CL_MEM_COPY_HOST_PTR,IntegerInputs)
        
        # Make the kernel and assign buffers
        self.JDkernel = program.get_kernel('doJorgensenDruedahlFix')
        self.JDkernel.set_args(self.mLvlData_buf,
                      self.MedShkData_buf,
                      self.ValueData_buf,
                      self.xLvlData_buf,
                      self.mGridDense_buf,
                      self.ShkGridDense_buf,
                      self.xLvlOut_buf,
                      self.ValueOut_buf,
                      self.IntegerInputs_buf)
        
    def __call__(self,mLvlData,MedShkData,ValueData,xLvlData,mGridDense,ShkGridDense):
        '''
        Use the Jorgensen-Druedahl convexity fix for given data.
        '''
        # Make arrays to hold the output
        bad_value = 0.0
        even_worse_value = -1e10
        xLvlOut = np.tile(np.reshape(mGridDense,(self.mGridDenseSize,1)),(1,self.ShkGridDenseSize)).flatten() # Spend all as a default
        ValueOut = bad_value*np.ones_like(xLvlOut)
        
        # Process the spending and value data just a bit
        ValueData_temp = ValueData.flatten()
        these = np.isnan(ValueData_temp)
        ValueData_temp[these] = even_worse_value
        xLvlData_temp = xLvlData.flatten()
        xLvlData_temp[these] = 0.0
        
        # Assign data to buffers
        queue.write_buffer(self.mLvlData_buf,mLvlData.flatten())
        queue.write_buffer(self.MedShkData_buf,MedShkData.flatten())
        queue.write_buffer(self.ValueData_buf,ValueData_temp)
        queue.write_buffer(self.xLvlData_buf,xLvlData_temp)
        queue.write_buffer(self.mGridDense_buf,mGridDense)
        queue.write_buffer(self.ShkGridDense_buf,ShkGridDense)
        queue.write_buffer(self.xLvlOut_buf,xLvlOut)
        queue.write_buffer(self.ValueOut_buf,ValueOut)
        
        self.JDkernel.set_args(self.mLvlData_buf,
                      self.MedShkData_buf,
                      self.ValueData_buf,
                      self.xLvlData_buf,
                      self.mGridDense_buf,
                      self.ShkGridDense_buf,
                      self.xLvlOut_buf,
                      self.ValueOut_buf,
                      self.IntegerInputs_buf)
        
        # Run the kernel and unpack the output
        queue.execute_kernel(self.JDkernel, [16*(self.ThreadCount/16 + 1)], [16])
        queue.read_buffer(self.xLvlOut_buf,xLvlOut)
        queue.read_buffer(self.ValueOut_buf,ValueOut)
    
        # Transform xLvlOut into a BilinearInterp and return it
        xLvlNow = np.concatenate((np.zeros((1,self.ShkGridDenseSize)),np.reshape(xLvlOut,(self.mGridDenseSize,self.ShkGridDenseSize))),axis=0)
        xFunc_this_pLvl = BilinearInterp(xLvlNow,np.insert(mGridDense,0,0.0),ShkGridDense)
        return xFunc_this_pLvl
        


def makeJDxLvlLayer(mLvlData,MedShkData,ValueData,xLvlData,mGridDense,ShkGridDense):
    '''
    Makes one permanent income "layer" of the expenditure function xFunc using
    the Jorgensen-Druedahl fix for non-concave problems with EGM.
    
    Parameters
    ----------
    mLvlData : np.array
        2D array of market resource levels generated through the EGM.
    MedShkData : np.array
        2D array of medical shocks generated through the EGM (exogenous here).
    ValueData : np.array
        2D array of (pseudo-inverse) values corresponding to the (mLvl,MedShk) values above.
    xLvlData : np.array
        2D array of expenditure levels generated through the EGM.
    mGridDense : np.array
        1D array of exogenous market resources levels for output.
    ShkGridDense : np.array
        1D array of exogenous medical shocks for output.
        
    Returns
    -------
    xFunc_this_pLvl : BilinearInterp
        Expenditure function for this permanent income level, represented as a
        bilinear interpolation over the exogenous "dense" grids.
    '''
    bad_value = 0.0
    even_worse_value = -1e10
    
    # Make integer inputs
    mLvlDataDim = mLvlData.shape[0]
    MedShkDataDim = mLvlData.shape[1]
    mGridDenseSize = mGridDense.size
    ShkGridDenseSize = ShkGridDense.size
    ThreadCount = mGridDenseSize*ShkGridDenseSize
    IntegerInputs = np.array([mLvlDataDim,MedShkDataDim,mGridDenseSize,ShkGridDenseSize,ThreadCount],dtype=np.int32)
    
    # Make arrays to hold the output
    xLvlOut = np.tile(np.reshape(mGridDense,(mGridDenseSize,1)),(1,ShkGridDenseSize)).flatten() # Spend all as a default
    ValueOut = bad_value*np.ones_like(xLvlOut)
    
    # Process the spending and value data just a bit
    ValueData_temp = ValueData.flatten()
    these = np.isnan(ValueData_temp)
    ValueData_temp[these] = even_worse_value
    xLvlData_temp = xLvlData.flatten()
    xLvlData_temp[these] = 0.0
    
    # Make buffers
    mLvlData_buf = ctx.create_buffer(cl.CL_MEM_READ_ONLY | cl.CL_MEM_COPY_HOST_PTR,mLvlData.flatten())
    MedShkData_buf = ctx.create_buffer(cl.CL_MEM_READ_ONLY | cl.CL_MEM_COPY_HOST_PTR,MedShkData.flatten())
    ValueData_buf = ctx.create_buffer(cl.CL_MEM_READ_ONLY | cl.CL_MEM_COPY_HOST_PTR,ValueData_temp)
    xLvlData_buf = ctx.create_buffer(cl.CL_MEM_READ_ONLY | cl.CL_MEM_COPY_HOST_PTR,xLvlData_temp)
    mGridDense_buf = ctx.create_buffer(cl.CL_MEM_READ_ONLY | cl.CL_MEM_COPY_HOST_PTR,mGridDense)
    ShkGridDense_buf = ctx.create_buffer(cl.CL_MEM_READ_ONLY | cl.CL_MEM_COPY_HOST_PTR,ShkGridDense)
    xLvlOut_buf = ctx.create_buffer(cl.CL_MEM_READ_WRITE | cl.CL_MEM_COPY_HOST_PTR,xLvlOut)
    ValueOut_buf = ctx.create_buffer(cl.CL_MEM_READ_WRITE | cl.CL_MEM_COPY_HOST_PTR,ValueOut)
    IntegerInputs_buf = ctx.create_buffer(cl.CL_MEM_READ_ONLY | cl.CL_MEM_COPY_HOST_PTR,IntegerInputs)
    
    # Assign buffers to the kernel
    JDkernel.set_args(mLvlData_buf,
                      MedShkData_buf,
                      ValueData_buf,
                      xLvlData_buf,
                      mGridDense_buf,
                      ShkGridDense_buf,
                      xLvlOut_buf,
                      ValueOut_buf,
                      IntegerInputs_buf)
    
    # Run the kernel and unpack the output
    queue.execute_kernel(JDkernel, [16*(ThreadCount/16 + 1)], [16])
    queue.read_buffer(xLvlOut_buf,xLvlOut)
    queue.read_buffer(ValueOut_buf,ValueOut)

    # Transform xLvlOut into a BilinearInterp and return it
    xLvlNow = np.concatenate((np.zeros((1,ShkGridDenseSize)),np.reshape(xLvlOut,(mGridDenseSize,ShkGridDenseSize))),axis=0)
    xFunc_this_pLvl = BilinearInterp(xLvlNow,np.insert(mGridDense,0,0.0),ShkGridDense)
    return xFunc_this_pLvl
    

def makeGridDenser(grid,factor):
    '''
    Makes an orderd 1D array denser by adding several points in between each value.
    
    Parameters
    ----------
    grid : np.array
        1D input array to be made denser.
    factor : int
        Factor by which to make input array denser.
        
    Returns
    -------
    grid_out : np.array
        Denser version of input array, with size grid.size*factor.
    '''
    grid_out = np.concatenate([np.linspace(grid[i],grid[i+1],num=factor,endpoint=False) for i in range(grid.size-1)]+[[grid[-1]]])
    return grid_out
