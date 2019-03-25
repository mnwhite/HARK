'''
This module calls OpenCL code to run the Jorgensen-Druedahl convexity fix for
the DynInsSel model.
'''

import sys
import os
import numpy as np
import opencl4py as cl
os.environ["PYOPENCL_CTX"] = "0:1" # This is where you choose a device number
sys.path.insert(0, os.path.abspath('../'))
sys.path.insert(0,'../../')
sys.path.insert(0,'../../ConsumptionSaving/')
sys.path.insert(0, os.path.abspath('./'))
from HARKinterpolation import LinearInterp, BilinearInterp, LinearInterpOnInterp1D
from HARKutilities import CRRAutility, CRRAutility_inv
from ConsPersistentShockModel import ValueFunc2D

f = open('JorgensenDruedahl.cl')
program_code = f.read()
f.close()

# Make a context and kernel
platforms = cl.Platforms()
ctx = platforms.create_some_context()
queue = ctx.create_queue(ctx.devices[0])
program = ctx.create_program(program_code)
JDkernel = program.get_kernel('doJorgensenDruedahlFix')
JDkernel_simple = program.get_kernel('doJorgensenDruedahlSimpleFix')

class JDfixer(object):
    '''
    A class-oriented implementation of the Jorgensen-Druedahl convexity fix for
    the DynInsSel model.  Each period, the solver creates an instance of this
    class, which is called whenever a JD fix is needed, passing current data
    to the buffers.
    '''
    def __init__(self,mLvlDataDim,DevDataDim,mGridDenseSize,DevGridDenseSize):
        '''
        Make a new JDfixer object
        '''
        self.mLvlDataDim = mLvlDataDim
        self.DevDataDim = DevDataDim
        self.mGridDenseSize = mGridDenseSize
        self.DevGridDenseSize = DevGridDenseSize
        self.ThreadCount = mGridDenseSize*DevGridDenseSize
        IntegerInputs = np.array([mLvlDataDim,DevDataDim,mGridDenseSize,DevGridDenseSize,self.ThreadCount],dtype=np.int32)
        data_temp = np.zeros(mLvlDataDim*DevDataDim)
        out_temp = np.zeros(self.ThreadCount)
        
        # Make buffers
        self.mLvlData_buf = ctx.create_buffer(cl.CL_MEM_READ_WRITE | cl.CL_MEM_COPY_HOST_PTR,data_temp)
        self.DevData_buf = ctx.create_buffer(cl.CL_MEM_READ_WRITE | cl.CL_MEM_COPY_HOST_PTR,data_temp)
        self.ValueData_buf = ctx.create_buffer(cl.CL_MEM_READ_WRITE | cl.CL_MEM_COPY_HOST_PTR,data_temp)
        self.xLvlData_buf = ctx.create_buffer(cl.CL_MEM_READ_WRITE | cl.CL_MEM_COPY_HOST_PTR,data_temp)
        self.mGridDense_buf = ctx.create_buffer(cl.CL_MEM_READ_WRITE | cl.CL_MEM_COPY_HOST_PTR,np.zeros(mGridDenseSize))
        self.DevGridDense_buf = ctx.create_buffer(cl.CL_MEM_READ_WRITE | cl.CL_MEM_COPY_HOST_PTR,np.zeros(DevGridDenseSize))
        self.xLvlOut_buf = ctx.create_buffer(cl.CL_MEM_READ_WRITE | cl.CL_MEM_COPY_HOST_PTR,out_temp)
        self.ValueOut_buf = ctx.create_buffer(cl.CL_MEM_READ_WRITE | cl.CL_MEM_COPY_HOST_PTR,out_temp)
        self.IntegerInputs_buf = ctx.create_buffer(cl.CL_MEM_READ_ONLY | cl.CL_MEM_COPY_HOST_PTR,IntegerInputs)
        
        # Make the kernel and assign buffers
        self.JDkernel = program.get_kernel('doJorgensenDruedahlFix')
        self.JDkernel.set_args(self.mLvlData_buf,
                      self.DevData_buf,
                      self.ValueData_buf,
                      self.xLvlData_buf,
                      self.mGridDense_buf,
                      self.DevGridDense_buf,
                      self.xLvlOut_buf,
                      self.ValueOut_buf,
                      self.IntegerInputs_buf)
        
    def __call__(self,mLvlData,DevData,ValueData,xLvlData,mGridDense,DevGridDense):
        '''
        Use the Jorgensen-Druedahl convexity fix for given data.
        '''
        # Make arrays to hold the output
        bad_value = 0.0
        even_worse_value = -1e10
        xLvlOut = np.tile(np.reshape(mGridDense,(self.mGridDenseSize,1)),(1,self.DevGridDenseSize)).flatten() # Spend all as a default
        ValueOut = bad_value*np.ones_like(xLvlOut)
        
        # Process the spending and value data just a bit
        ValueData_temp = ValueData.flatten()
        these = np.isnan(ValueData_temp)
        ValueData_temp[these] = even_worse_value
        xLvlData_temp = xLvlData.flatten()
        xLvlData_temp[these] = 0.0
        
        # Assign data to buffers
        queue.write_buffer(self.mLvlData_buf,mLvlData.flatten())
        queue.write_buffer(self.DevData_buf,DevData.flatten())
        queue.write_buffer(self.ValueData_buf,ValueData_temp)
        queue.write_buffer(self.xLvlData_buf,xLvlData_temp)
        queue.write_buffer(self.mGridDense_buf,mGridDense)
        queue.write_buffer(self.DevGridDense_buf,DevGridDense)
        queue.write_buffer(self.xLvlOut_buf,xLvlOut)
        queue.write_buffer(self.ValueOut_buf,ValueOut)
        
        # Run the kernel and unpack the output
        queue.execute_kernel(self.JDkernel, [16*(self.ThreadCount/16 + 1)], [16])
        queue.read_buffer(self.xLvlOut_buf,xLvlOut)
        queue.read_buffer(self.ValueOut_buf,ValueOut)
        
        if np.sum(np.isnan(xLvlOut)) > 0.:
            print('Found some NaNs!')
    
        # Transform xLvlOut into a BilinearInterp and return it
        xLvlOut_reshaped = np.reshape(xLvlOut,(self.mGridDenseSize,self.DevGridDenseSize))
        xLvlNow = np.concatenate((np.zeros((1,self.DevGridDenseSize)),xLvlOut_reshaped),axis=0)
        xFunc_this_pLvl = BilinearInterp(xLvlNow,np.insert(mGridDense,0,0.0),DevGridDense)
        return xFunc_this_pLvl, (xLvlOut_reshaped[:,np.arange(0,self.DevGridDenseSize,3)]).transpose()
    
    
    
class JDfixerSimple(object):
    '''
    A class-oriented implementation of the Jorgensen-Druedahl convexity fix for
    the DynInsSel model.  Each period, the solver creates an instance of this
    class, which is called whenever a JD fix is needed, passing current data
    to the buffers.  This version is used for the MedShk=0 case, so the fix is
    over (mLvl, pLvl) rather than (mLvl, Dev) within a pLvl.
    '''
    def __init__(self,mLvlDataDim,pLvlDataDim,mGridDenseSize,pGridDenseSize,CRRA):
        '''
        Make a new JDfixer object
        '''
        self.mLvlDataDim = mLvlDataDim
        self.pLvlDataDim = pLvlDataDim
        self.mGridDenseSize = mGridDenseSize
        self.pGridDenseSize = pGridDenseSize
        self.ThreadCount = mGridDenseSize*pGridDenseSize
        self.CRRA = CRRA
        IntegerInputs = np.array([mLvlDataDim,pLvlDataDim,mGridDenseSize,pGridDenseSize,self.ThreadCount],dtype=np.int32)
        data_temp = np.zeros(mLvlDataDim*pLvlDataDim)
        out_temp = np.zeros(self.ThreadCount)
        
        # Make buffers
        self.mLvlData_buf = ctx.create_buffer(cl.CL_MEM_READ_WRITE | cl.CL_MEM_COPY_HOST_PTR,data_temp)
        self.pLvlData_buf = ctx.create_buffer(cl.CL_MEM_READ_WRITE | cl.CL_MEM_COPY_HOST_PTR,data_temp)
        self.ValueData_buf = ctx.create_buffer(cl.CL_MEM_READ_WRITE | cl.CL_MEM_COPY_HOST_PTR,data_temp)
        self.cLvlData_buf = ctx.create_buffer(cl.CL_MEM_READ_WRITE | cl.CL_MEM_COPY_HOST_PTR,data_temp)
        self.mGridDense_buf = ctx.create_buffer(cl.CL_MEM_READ_WRITE | cl.CL_MEM_COPY_HOST_PTR,np.zeros(mGridDenseSize))
        self.pGridDense_buf = ctx.create_buffer(cl.CL_MEM_READ_WRITE | cl.CL_MEM_COPY_HOST_PTR,np.zeros(pGridDenseSize))
        self.cLvlOut_buf = ctx.create_buffer(cl.CL_MEM_READ_WRITE | cl.CL_MEM_COPY_HOST_PTR,out_temp)
        self.ValueOut_buf = ctx.create_buffer(cl.CL_MEM_READ_WRITE | cl.CL_MEM_COPY_HOST_PTR,out_temp)
        self.IntegerInputs_buf = ctx.create_buffer(cl.CL_MEM_READ_ONLY | cl.CL_MEM_COPY_HOST_PTR,IntegerInputs)
        
        # Make the kernel and assign buffers
        self.JDkernel = program.get_kernel('doJorgensenDruedahlSimpleFix')
        self.JDkernel.set_args(self.mLvlData_buf,
                      self.pLvlData_buf,
                      self.ValueData_buf,
                      self.cLvlData_buf,
                      self.mGridDense_buf,
                      self.pGridDense_buf,
                      self.cLvlOut_buf,
                      self.ValueOut_buf,
                      self.IntegerInputs_buf)
        
    def __call__(self,mLvlData,pLvlData,ValueData,cLvlData,mGridDense,pGridDense,EndOfPrdvFunc_Cnst):
        '''
        Use the Jorgensen-Druedahl convexity fix for given data.
        '''
        u = lambda x : CRRAutility(x,gam=self.CRRA)
        uinv = lambda x : CRRAutility_inv(x,gam=self.CRRA)
        
        # Construct the grid of mLvl that will come out of JD fix
        mGrid_tiled = np.tile(np.reshape(mGridDense,(self.mGridDenseSize,1)),(1,self.pGridDenseSize))
        pGrid_tiled = np.tile(np.reshape(pGridDense,(1,self.pGridDenseSize)),(self.mGridDenseSize,1))
        mLvlOut = mGrid_tiled*pGrid_tiled
        if pGridDense[0] == 0.:
            mLvlOut[:,0] = mLvlOut[:,1]
        
        # Make arrays to hold the output
        ValueOut = (uinv(u(mLvlOut) + EndOfPrdvFunc_Cnst(pGrid_tiled))).flatten() # Consume all as default
        cLvlOut = mLvlOut.flatten()
        
        # Process the spending and value data just a bit
        even_worse_value = -1e10
        ValueData_temp = ValueData.flatten()
        these = np.isnan(ValueData_temp)
        ValueData_temp[these] = even_worse_value
        cLvlData_temp = cLvlData.flatten()
        cLvlData_temp[these] = 0.0
        
        # Assign data to buffers
        queue.write_buffer(self.mLvlData_buf,mLvlData.flatten())
        queue.write_buffer(self.pLvlData_buf,pLvlData.flatten())
        queue.write_buffer(self.ValueData_buf,ValueData_temp)
        queue.write_buffer(self.cLvlData_buf,cLvlData_temp)
        queue.write_buffer(self.mGridDense_buf,mGridDense)
        queue.write_buffer(self.pGridDense_buf,pGridDense)
        queue.write_buffer(self.cLvlOut_buf,cLvlOut)
        queue.write_buffer(self.ValueOut_buf,ValueOut)
        
        # Run the kernel and unpack the output
        queue.execute_kernel(self.JDkernel, [16*(self.ThreadCount/16 + 1)], [16])
        queue.read_buffer(self.cLvlOut_buf,cLvlOut)
        queue.read_buffer(self.ValueOut_buf,ValueOut)
    
        # Transform cLvlOut into a BilinearInterp and return it
        mLvlNow = np.concatenate((np.zeros((1,self.pGridDenseSize)),np.reshape(mLvlOut,(self.mGridDenseSize,self.pGridDenseSize))),axis=0)
        cLvlNow = np.concatenate((np.zeros((1,self.pGridDenseSize)),np.reshape(cLvlOut,(self.mGridDenseSize,self.pGridDenseSize))),axis=0)
        vNvrsNow = np.concatenate((np.zeros((1,self.pGridDenseSize)),np.reshape(ValueOut,(self.mGridDenseSize,self.pGridDenseSize))),axis=0)
        cFunc_by_pLvl_ZeroShk = []
        vNvrsFunc_by_pLvl_ZeroShk = []
        for j in range(self.pGridDenseSize):
            mLvl_temp = mLvlNow[:,j]
            cLvl_temp = cLvlNow[:,j]
            vNvrs_temp= vNvrsNow[:,j]
            cFunc_by_pLvl_ZeroShk.append(LinearInterp(mLvl_temp,cLvl_temp))
            vNvrsFunc_by_pLvl_ZeroShk.append(LinearInterp(mLvl_temp,vNvrs_temp))
        cFuncZeroShk = LinearInterpOnInterp1D(cFunc_by_pLvl_ZeroShk,pGridDense)
        vNvrsFuncZeroShk = LinearInterpOnInterp1D(vNvrsFunc_by_pLvl_ZeroShk,pGridDense)
        vFuncZeroShk = ValueFunc2D(vNvrsFuncZeroShk, self.CRRA)
        
        return cFuncZeroShk, vFuncZeroShk
        
    

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
