'''
This is an attempt to translate the wonky value function over mLvl,pLvl,Dev into OpenCL.
'''

import sys
import os
import numpy as np
import opencl4py as cl
os.environ["PYOPENCL_CTX"] = "0:1" # This is where you choose a device number
sys.path.insert(0, os.path.abspath('../'))
sys.path.insert(0,'../../')
sys.path.insert(0, os.path.abspath('./'))

f = open('ValueFunc.cl')
program_code = f.read()
f.close()

# Make a context and kernel
platforms = cl.Platforms()
ctx = platforms.create_some_context()
queue = ctx.create_queue(ctx.devices[0])
program = ctx.create_program(program_code)
vFuncKernel = program.get_kernel('evalValueFunc')

class ValueFuncCL(object):
    '''
    A class for representing the health-coinsurance rate conditional value function
    over (mLvl,pLvl,Dev) using OpenCL to accelerate evaluation.
    
    Parameters
    ----------
    mNrmGrid : np.array
        Array of normalized market resource values.
    pLvlGrid : np.array
        Array of permanent income levels.
    pLvlGridAlt : np.array
        Alternate array of permanent income levels, used only for determining mLvlGrid by pLvl.
    vNvrsZeroShkData : np.array
        2D array of pseudo-inverse value when MedShk=0 of shape (pLvlGrid.size,mNrmGrid.size+1).
    vNvrsRescaledData : np.array
        3D array of rescaled pseudo-inverse value of shape (DevCount,pLvlGrid.size,mNrmGrid.size).
    CRRA : float
        Coefficient of relative risk aversion.
    DevMin : float
        Lower bound of DevGrid.
    DevMax : float
        Upper bound of DevGrid.
    DevCount : int
        Number of points in DevGrid.
        
    Returns
    -------
    None
    '''
    def __init__(self,mNrmGrid,pLvlGrid,pLvlGridAlt,vNvrsZeroShkData,vNvrsRescaledData,CRRA,DevMin,DevMax,DevCount):
        # Make two short vectors of inputs
        self.IntegerInputs = np.array([mNrmGrid.size,pLvlGrid.size,DevCount,0],dtype=np.int32) # Last element will be overwritten
        DoubleInputs = np.array([CRRA,DevMin,DevMax])
        
        # Make buffers
        self.mNrmGrid_buf = ctx.create_buffer(cl.CL_MEM_READ_ONLY | cl.CL_MEM_COPY_HOST_PTR,mNrmGrid)
        self.pLvlGrid_buf = ctx.create_buffer(cl.CL_MEM_READ_ONLY | cl.CL_MEM_COPY_HOST_PTR,pLvlGrid)
        self.pLvlGridAlt_buf = ctx.create_buffer(cl.CL_MEM_READ_ONLY | cl.CL_MEM_COPY_HOST_PTR,pLvlGridAlt)
        self.vNvrsZeroShkData_buf = ctx.create_buffer(cl.CL_MEM_READ_ONLY | cl.CL_MEM_COPY_HOST_PTR,vNvrsZeroShkData)
        self.vNvrsRescaledData_buf = ctx.create_buffer(cl.CL_MEM_READ_ONLY | cl.CL_MEM_COPY_HOST_PTR,vNvrsRescaledData)
        self.DoubleInputs_buf = ctx.create_buffer(cl.CL_MEM_READ_ONLY | cl.CL_MEM_COPY_HOST_PTR,DoubleInputs)
        self.IntegerInputs_buf = ctx.create_buffer(cl.CL_MEM_READ_ONLY | cl.CL_MEM_COPY_HOST_PTR,self.IntegerInputs)
 
    def __call__(self,mLvl,pLvl,Dev):
        '''
        Evaluate the value function at a set of query points.
        
        Parameters
        ----------
        mLvl : np.array
            Array of market resource query points.
        pLvl : np.array
            Array of permanent income query points.
        Dev : np.array
            Array of medical shock deviation query points.
            
        Returns
        -------
        v : np.array
            Array of values at all query points.
        '''
        orig_shape = mLvl.shape
        N = mLvl.size # Number of query points
        self.IntegerInputs[3] = N
        queue.write_buffer(self.IntegerInputs_buf,self.IntegerInputs)
        ValueOut = np.zeros(N)
        
        if N > 0:
            # Make query and return buffers
            mLvlQuery_buf = ctx.create_buffer(cl.CL_MEM_READ_ONLY | cl.CL_MEM_COPY_HOST_PTR,mLvl.flatten())
            pLvlQuery_buf = ctx.create_buffer(cl.CL_MEM_READ_ONLY | cl.CL_MEM_COPY_HOST_PTR,pLvl.flatten())
            DevQuery_buf  = ctx.create_buffer(cl.CL_MEM_READ_ONLY | cl.CL_MEM_COPY_HOST_PTR,Dev.flatten())
            ValueOut_buf  = ctx.create_buffer(cl.CL_MEM_READ_ONLY | cl.CL_MEM_COPY_HOST_PTR,ValueOut)
            
            # Assign buffers to kernel arguments
            vFuncKernel.set_args(self.mNrmGrid_buf,
                                 self.pLvlGrid_buf,
                                 self.pLvlGridAlt_buf,
                                 self.vNvrsZeroShkData_buf,
                                 self.vNvrsRescaledData_buf,
                                 mLvlQuery_buf,
                                 pLvlQuery_buf,
                                 DevQuery_buf,
                                 ValueOut_buf,
                                 self.DoubleInputs_buf,
                                 self.IntegerInputs_buf)
            
            # Execute the kernel, extract and reshape the output
            queue.execute_kernel(vFuncKernel, [16*(N/16 + 1)], [16])
            queue.read_buffer(ValueOut_buf,ValueOut)
            v = np.reshape(ValueOut, orig_shape)
        else:
            v = np.array([]) # Trivial case
            
        return v
    
    
    def dumpCLdata(self):
        '''
        Copy CL buffer data for mNrmGrid, pLvlGrid, pLvlGridAlt, vNvrsZeroShk,
        and vNvrsRescaled into arrays that are attributes of self.
        '''
        self.mNrmGrid = np.zeros(self.IntegerInputs[0])
        self.pLvlGrid = np.zeros(self.IntegerInputs[1])
        self.pLvlGridAlt = np.zeros((self.IntegerInputs[1]))
        self.vNvrsZeroShk = np.zeros((self.IntegerInputs[1],self.IntegerInputs[0]+1))
        self.vNvrsRescaled = np.zeros((self.IntegerInputs[2],self.IntegerInputs[1],self.IntegerInputs[0]))
        
        queue.read_buffer(self.mNrmGrid_buf,self.mNrmGrid)
        queue.read_buffer(self.pLvlGrid_buf,self.pLvlGrid)
        queue.read_buffer(self.pLvlGridAlt_buf,self.pLvlGridAlt)
        queue.read_buffer(self.vNvrsZeroShkData_buf,self.vNvrsZeroShk)
        queue.read_buffer(self.vNvrsRescaledData_buf,self.vNvrsRescaled)
        