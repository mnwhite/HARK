'''
This module tests an inline function that acts like np.searchsorted.
'''

import sys
import os
import numpy as np
import opencl4py as cl
os.environ["PYOPENCL_CTX"] = "0:1" # This is where you choose a device number
sys.path.insert(0, os.path.abspath('../'))
sys.path.insert(0,'../../')
sys.path.insert(0, os.path.abspath('./'))

f = open('JDfix3Dalt.cl')
program_code = f.read()
f.close()

# Make a context and kernel
platforms = cl.Platforms()
ctx = platforms.create_some_context()
queue = ctx.create_queue(ctx.devices[0])
program = ctx.create_program(program_code)

# Make some test buffers
ArrayToSearch = np.linspace(2.,57.,88)
DataArray = np.random.rand(1000)*60
IntegerInputs = np.array([0,ArrayToSearch.size-1,DataArray.size],dtype=np.int32)
ArrayToSearch_buf = ctx.create_buffer(cl.CL_MEM_READ_ONLY | cl.CL_MEM_COPY_HOST_PTR,ArrayToSearch)
DataArray_buf = ctx.create_buffer(cl.CL_MEM_READ_ONLY | cl.CL_MEM_COPY_HOST_PTR,DataArray)
IdxArray_buf = ctx.create_buffer(cl.CL_MEM_READ_WRITE | cl.CL_MEM_COPY_HOST_PTR,np.zeros_like(DataArray,dtype=int))
IntegerInputs_buf = ctx.create_buffer(cl.CL_MEM_READ_ONLY | cl.CL_MEM_COPY_HOST_PTR,IntegerInputs)

# Define the search test kernel
SearchKernel = program.get_kernel('searchTest')
SearchKernel.set_args(ArrayToSearch_buf,
                      DataArray_buf,
                      IdxArray_buf,
                      IntegerInputs_buf)

# Execute the search test kernel
IdxOut = np.zeros(DataArray.size,dtype=int)
queue.execute_kernel(SearchKernel, [16*(DataArray.size/16 + 1)], [16])
queue.read_buffer(IdxArray_buf,IdxOut)

# Run the search using numpy
IdxTrue = np.searchsorted(ArrayToSearch,DataArray)


