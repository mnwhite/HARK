'''
This module calls OpenCL code to run the Jorgensen-Druedahl convexity fix for
the health investment model.
'''

import sys
import os
import numpy as np
import opencl4py as cl
#from copy import copy
os.environ["PYOPENCL_CTX"] = "1:1" # This is where you choose a device number
sys.path.insert(0, os.path.abspath('../'))
sys.path.insert(0,'../../')
sys.path.insert(0, os.path.abspath('./'))

ALT = True

if ALT:
    f = open('JDfix3Dalt.cl')
else:
    f = open('JDfix3D.cl')
program_code = f.read()
f.close()

# Make a context and kernel
platforms = cl.Platforms()
ctx = platforms.create_some_context()
queue = ctx.create_queue(ctx.devices[0])
program = ctx.create_program(program_code)

class JDfixer(object):
    '''
    A class-oriented implementation of the Jorgensen-Druedahl convexity fix for
    the health investment model.  Each period, the solver creates an instance of
    this class, which is called whenever a JD fix is needed, passing current data
    to the buffers.
    '''
    def __init__(self,bLvlDataDim,hLvlDataDim,MedShkDataDim,bGridDenseSize,hGridDenseSize,ShkGridDenseSize):
        '''
        Make a new JDfixer object
        '''
        self.bLvlDataDim = bLvlDataDim
        self.hLvlDataDim = hLvlDataDim
        self.MedShkDataDim = MedShkDataDim
        self.bGridDenseSize = bGridDenseSize
        self.hGridDenseSize = hGridDenseSize
        self.ShkGridDenseSize = ShkGridDenseSize
        DenseCount = bGridDenseSize*hGridDenseSize*ShkGridDenseSize
        if ALT:
            self.ThreadCount = MedShkDataDim-1
        else:
            self.ThreadCount = DenseCount
        IntegerInputs = np.array([bLvlDataDim,hLvlDataDim,MedShkDataDim,bGridDenseSize,hGridDenseSize,ShkGridDenseSize,self.ThreadCount],dtype=np.int32)
        data_temp = np.zeros(bLvlDataDim*hLvlDataDim*MedShkDataDim)
        out_temp = np.zeros(DenseCount)
        self.data_size = data_temp.size
        
        # Make buffers
        self.bLvlData_buf = ctx.create_buffer(cl.CL_MEM_READ_ONLY | cl.CL_MEM_COPY_HOST_PTR,data_temp)
        self.hLvlData_buf = ctx.create_buffer(cl.CL_MEM_READ_ONLY | cl.CL_MEM_COPY_HOST_PTR,data_temp)
        self.MedShkData_buf = ctx.create_buffer(cl.CL_MEM_READ_ONLY | cl.CL_MEM_COPY_HOST_PTR,data_temp)
        self.ValueData_buf = ctx.create_buffer(cl.CL_MEM_READ_ONLY | cl.CL_MEM_COPY_HOST_PTR,data_temp)
        self.dvdhData_buf = ctx.create_buffer(cl.CL_MEM_READ_ONLY | cl.CL_MEM_COPY_HOST_PTR,data_temp)
        self.xLvlData_buf = ctx.create_buffer(cl.CL_MEM_READ_ONLY | cl.CL_MEM_COPY_HOST_PTR,data_temp)
        self.iLvlData_buf = ctx.create_buffer(cl.CL_MEM_READ_ONLY | cl.CL_MEM_COPY_HOST_PTR,data_temp)
        self.bGridDense_buf = ctx.create_buffer(cl.CL_MEM_READ_ONLY | cl.CL_MEM_COPY_HOST_PTR,np.zeros(bGridDenseSize))
        self.hGridDense_buf = ctx.create_buffer(cl.CL_MEM_READ_ONLY | cl.CL_MEM_COPY_HOST_PTR,np.zeros(hGridDenseSize))
        self.ShkGridDense_buf = ctx.create_buffer(cl.CL_MEM_READ_ONLY | cl.CL_MEM_COPY_HOST_PTR,np.zeros(ShkGridDenseSize))
        self.xLvlOut_buf = ctx.create_buffer(cl.CL_MEM_READ_WRITE | cl.CL_MEM_COPY_HOST_PTR,out_temp)
        self.iLvlOut_buf = ctx.create_buffer(cl.CL_MEM_READ_WRITE | cl.CL_MEM_COPY_HOST_PTR,out_temp)
        self.ValueOut_buf = ctx.create_buffer(cl.CL_MEM_READ_WRITE | cl.CL_MEM_COPY_HOST_PTR,out_temp)
        self.dvdhOut_buf = ctx.create_buffer(cl.CL_MEM_READ_WRITE | cl.CL_MEM_COPY_HOST_PTR,out_temp)
        self.IntegerInputs_buf = ctx.create_buffer(cl.CL_MEM_READ_ONLY | cl.CL_MEM_COPY_HOST_PTR,IntegerInputs)
        
        # Make the kernel and assign buffers
        self.JDkernel = program.get_kernel('doJorgensenDruedahlFix3D')
        self.JDkernel.set_args(self.bLvlData_buf,
                      self.hLvlData_buf,
                      self.MedShkData_buf,
                      self.ValueData_buf,
                      self.dvdhData_buf,
                      self.xLvlData_buf,
                      self.iLvlData_buf,
                      self.bGridDense_buf,
                      self.hGridDense_buf,
                      self.ShkGridDense_buf,
                      self.xLvlOut_buf,
                      self.iLvlOut_buf,
                      self.ValueOut_buf,
                      self.dvdhOut_buf,
                      self.IntegerInputs_buf)
        
        
    def __call__(self,bLvlData,hLvlData,MedShkData,ValueData,dvdhData,xLvlData,iLvlData,bGridDense,hGridDense,ShkGridDense):
        '''
        Use the Jorgensen-Druedahl convexity fix for given data.
        '''
        # Make arrays to hold the output
        # Consider turning bad_value into value from spending all b on x.
        bad_value = -1e100
        even_worse_value = -1e10
        xLvlOut = np.tile(np.reshape(bGridDense,(self.bGridDenseSize,1,1)),(1,self.hGridDenseSize,self.ShkGridDenseSize)).flatten() # Spend all as a default
        iLvlOut = np.zeros_like(xLvlOut)
        ValueOut = bad_value*np.ones_like(xLvlOut)
        dvdhOut = np.zeros_like(xLvlOut)
        
        # Process the spending and value data just a bit
        ValueData_temp = ValueData.flatten()
        these = np.isnan(ValueData_temp)
        ValueData_temp[these] = even_worse_value
        dvdhData_temp = dvdhData.flatten()
        dvdhData_temp[these] = 0.0
        xLvlData_temp = xLvlData.flatten()
        xLvlData_temp[these] = 0.0
        iLvlData_temp = iLvlData.flatten()
        iLvlData_temp[these] = 0.0
        
        # Assign data to buffers
        queue.write_buffer(self.bLvlData_buf,bLvlData.flatten())
        queue.write_buffer(self.hLvlData_buf,hLvlData.flatten())
        queue.write_buffer(self.MedShkData_buf,MedShkData.flatten())
        queue.write_buffer(self.ValueData_buf,ValueData_temp)
        queue.write_buffer(self.dvdhData_buf,dvdhData_temp)
        queue.write_buffer(self.xLvlData_buf,xLvlData_temp)
        queue.write_buffer(self.iLvlData_buf,iLvlData_temp)
        queue.write_buffer(self.bGridDense_buf,bGridDense)
        queue.write_buffer(self.hGridDense_buf,hGridDense)
        queue.write_buffer(self.ShkGridDense_buf,ShkGridDense)
        queue.write_buffer(self.xLvlOut_buf,xLvlOut)
        queue.write_buffer(self.iLvlOut_buf,iLvlOut)
        queue.write_buffer(self.ValueOut_buf,ValueOut)
        queue.write_buffer(self.dvdhOut_buf,dvdhOut)
        
        # Run the kernel and unpack the output
        queue.execute_kernel(self.JDkernel, [16*(self.ThreadCount/16 + 1)], [16])
        queue.read_buffer(self.xLvlOut_buf,xLvlOut)
        queue.read_buffer(self.iLvlOut_buf,iLvlOut)
        queue.read_buffer(self.ValueOut_buf,ValueOut)
        queue.read_buffer(self.dvdhOut_buf,dvdhOut)
        
        xLvlArray = np.reshape(xLvlOut,(self.bGridDenseSize,self.hGridDenseSize,self.ShkGridDenseSize))
        iLvlArray = np.reshape(iLvlOut,(self.bGridDenseSize,self.hGridDenseSize,self.ShkGridDenseSize))
        vNvrsArray = np.reshape(ValueOut,(self.bGridDenseSize,self.hGridDenseSize,self.ShkGridDenseSize))
        dvdhArray = np.reshape(dvdhOut,(self.bGridDenseSize,self.hGridDenseSize,self.ShkGridDenseSize))
        
        # Perform "error correction" on the output to fill in state points that are (somehow) missing
        for j in range(self.hGridDenseSize):
            for k in range(self.ShkGridDenseSize):
                errors = np.where(vNvrsArray[:,j,k] == bad_value)[0].tolist()
                #print(j,k,errors)
                while len(errors) > 0:
                    bot = errors[0]
                    i = 0
                    go = True
                    while go:
                        i += 1
                        errors.pop(0)
                        if len(errors) == 0:
                            go = False
                            continue
                        if errors[0] > bot + i:
                            go = False
                    top = bot + i
                    if top == self.bGridDenseSize:
                        lo = bot-2
                        hi = bot-1
                    else:
                        lo = bot-1
                        hi = top
                    bLo = bGridDense[lo]
                    bHi = bGridDense[hi]
                    alpha = (bGridDense[bot:top] - bLo)/(bHi - bLo)
                    alpha_comp = 1. - alpha
                    #print(bot,top,lo,hi,alpha)
                    vNvrsArray[bot:top,j,k] = alpha_comp*vNvrsArray[lo,j,k] + alpha*vNvrsArray[hi,j,k]
                    xLvlArray[bot:top,j,k] = alpha_comp*xLvlArray[lo,j,k] + alpha*xLvlArray[hi,j,k]
                    iLvlArray[bot:top,j,k] = alpha_comp*iLvlArray[lo,j,k] + alpha*iLvlArray[hi,j,k]
                    dvdhArray[bot:top,j,k] = alpha_comp*dvdhArray[lo,j,k] + alpha*dvdhArray[hi,j,k]
                        
    
        # Return results of the Jorgensen-Druedahl fix
        return xLvlArray, iLvlArray, vNvrsArray, dvdhArray
