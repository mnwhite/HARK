# pragma OPENCL EXTENSION cl_khr_fp64 : enable

/* Make 3x3 transformation matrix from four points in 3D space */
inline double16 make3DtransformationMatrix(
     double Ax
    ,double Ay
    ,double Az
    ,double Bx
    ,double By
    ,double Bz
    ,double Cx
    ,double Cy
    ,double Cz
    ,double Dx
    ,double Dy
    ,double Dz
    ) {
    double a11 = Ax - Dx;
    double a12 = Bx - Dx;
    double a13 = Cx - Dx;
    double a21 = Ay - Dy;
    double a22 = By - Dy;
    double a23 = Cy - Dy;
    double a31 = Az - Dz;
    double a32 = Bz - Dz;
    double a33 = Cz - Dz;
    double det = a11*a22*a33 + a21*a32*a13 + a31*a12*a23 - a11*a32*a23 - a31*a22*a13 - a21*a12*a33;
    double b11 = (a22*a33 - a23*a32)/det;
    double b12 = (a13*a32 - a12*a33)/det;
    double b13 = (a12*a23 - a13*a22)/det;
    double b21 = (a23*a31 - a21*a33)/det;
    double b22 = (a11*a33 - a13*a31)/det;
    double b23 = (a13*a21 - a11*a23)/det;
    double b31 = (a21*a32 - a22*a31)/det;
    double b32 = (a12*a31 - a11*a32)/det;
    double b33 = (a11*a22 - a12*a21)/det;
    double16 matrix_data = (double16)(b11,b12,b13,b21,b22,b23,b31,b32,b33,0.0,0.0,0.0,0.0,0.0,0.0,0.0);
    return matrix_data;
}


/* Calculate barycentric weights for a 3D point using a 3x3 transition matrix and a reference point */
inline double4 calcBarycentricWeights(
     double4 Data
    ,double4 Ref
    ,double4 T1
    ,double4 T2
    ,double4 T3) {
    
    double4 DataX = Data - Ref;
    double w1 = dot(T1,DataX);
    double w2 = dot(T2,DataX);
    double w3 = dot(T3,DataX);
    double w4 = 1.0 - w1 - w2 - w3;
    double4 weights = (double4)(w1,w2,w3,w4);
    return weights;
}


/* Find upper index of a number in an array */
inline int findIndex(
    __global double* ARRAY /* Array to search */
    ,int Start     /* Starting index of search */
    ,int End       /* Ending index of search */
    ,double xTarg  /* Target value to search for */
    ) {
    int jBot = Start;
    int jTop = End;
    double xBot = ARRAY[jBot];
    double xTop = ARRAY[jTop];
    int jDiff = jTop - jBot;
    int jNew = jBot + jDiff/2;
    double xNew = ARRAY[jNew];
    if (xTarg <= xBot) {
        return Start;
    }
    if (xTarg > xTop) {
        return End+1;
    }
    while (jDiff > 1) {
        if (xTarg < xNew) {
            jTop = jNew;
            xTop = xNew;
        }
        else {
            jBot = jNew;
            xBot = xNew;
        }
        jDiff = jTop - jBot;
        jNew = jBot + jDiff/2;
        xNew = ARRAY[jNew];
    }
    return jTop;
}
        

/* Test kernel to see if passing arrays to inline functions works */        
__kernel void searchTest(
     __global double *ArrayToSearch
    ,__global double *DataArray
    ,__global int *IdxArray
    ,__global int *IntegerInputs
    ) {
    int Idx0 = IntegerInputs[0];
    int Idx1 = IntegerInputs[1];
    int ThreadCount = IntegerInputs[2];    
        
    int Gid = get_global_id(0);     /* global thread id */
    if (Gid >= ThreadCount) {
        return;
    }
    
    double Data = DataArray[Gid];
    int IdxTarg = findIndex(ArrayToSearch,Idx0,Idx1,Data);
    IdxArray[Gid] = IdxTarg;
}
        
        
/* Kernel for implementing Jorgensen-Druedahl in the health investment model */
__kernel void doJorgensenDruedahlFix3D(
     __global double *bLvlData           /* data on endogenous bLvl */
    ,__global double *hLvlData           /* data on endogenous hLvl */ 
    ,__global double *MedShkData         /* data on medical shocks  */
    ,__global double *ValueData          /* data on value at (b,h,Shk) */
    ,__global double *dvdhData           /* data on marginal value at (b,h,Shk) */
    ,__global double *xLvlData           /* data on optimal xLvl    */
    ,__global double *iLvlData           /* data on optimal iLvl    */
    ,__global double *bGridDense         /* exogenous grid of bLvl  */
    ,__global double *hGridDense         /* exogenous grid of hLvl  */
    ,__global double *ShkGridDense       /* exogenous grid of MedShk */
    ,__global double *xLvlOut            /* J-D fixed xLvl to return */
    ,__global double *iLvlOut            /* J-D fixed iLvl to return */
    ,__global double *ValueOut           /* J-D fixed value to return */
    ,__global double *dvdhOut            /* J-D fixed marg value to return */
    ,__global int *IntegerInputs         /* integers that characterize problem */
    ) {
    
    /* Unpack the integer inputs */
    int bLvlDataDim = IntegerInputs[0];
    int hLvlDataDim = IntegerInputs[1];
    int MedShkDataDim = IntegerInputs[2];
    int bGridDenseSize = IntegerInputs[3];
    int hGridDenseSize = IntegerInputs[4];
    int ShkGridDenseSize = IntegerInputs[5];
    int ThreadCount = IntegerInputs[6];
    int hAndShkGridDenseSize = hGridDenseSize*ShkGridDenseSize;
    
    /* Initialize this thread's id and get this thread's constant (mLvl,MedShk) identity */
    int Gid = get_global_id(0);     /* global thread id */
    if (Gid >= ThreadCount) {
        return;
    }
    
    /* Get data on the minimum and maximum k index for the dense grid of medical shocks */
    double ShkMin = MedShkData[Gid];
    double ShkMax = MedShkData[Gid+1];
    int kData = Gid;
    int kMin = findIndex(ShkGridDense,0,ShkGridDenseSize-1,ShkMin);
    int kMax = findIndex(ShkGridDense,0,ShkGridDenseSize-1,ShkMax);
    if (kData == (ThreadCount-1)) { /* Hacky fix for top MedShk data layer */
        kMax = ShkGridDenseSize;
    }
    
    /* Initialize some variables for use in the main loop */
    double bLvl;
    double hLvl;
    double MedShk;
    double Value;
    double iLvl;
    double xLvl;
    double dvdh;
    double b1;
    double b2;
    double b3;
    double b4;
    double h1;
    double h2;
    double h3;
    double h4;
    double Shk1;
    double Shk2;
    double Shk3;
    double Shk4;
    double v1;
    double v2;
    double v3;
    double v4;
    double dvdh1;
    double dvdh2;
    double dvdh3;
    double dvdh4;
    double x1;
    double x2;
    double x3;
    double x4;
    double i1;
    double i2;
    double i3;
    double i4;
    double bMin;
    double bMax;
    double hMin;
    double hMax;
    double vNew;
    double xNew;
    double iNew;
    double dvdhNew;
    double4 SectorWeights = (double4)(0.0,0.0,0.0,0.0);
    int IdxA;
    int IdxB;
    int IdxC;
    int IdxD;
    int IdxE;
    int IdxF;
    int IdxG;
    int IdxH;
    int Idx1;
    int Idx2;
    int Idx3;
    int Idx4;
    int iData;
    int jData;
    int iDense;
    int jDense;
    int kDense;
    int iMin;
    int iMax;
    int jMin;
    int jMax;
    int DenseIdx;
    double16 TransMatrix;
    double4 T_row1;
    double4 T_row2;
    double4 T_row3;
    double4 RefPoint; 
    double4 DensePoint;
    double4 v_data;
    double4 x_data;
    double4 i_data;
    double4 dvdh_data;
    int InTop;
    int InBot;
        
    /* Loop over bLvl-hLvl data sectors for this kData */
    double tol = 0.01;
    double4 Low = (double4)(-tol,-tol,-tol,-tol);
    double4 High = (double4)(1.0+tol,1.0+tol,1.0+tol,1.0+tol);
    int hAndShkDataDim = hLvlDataDim*MedShkDataDim;
    for (iData=0; iData < (bLvlDataDim-1); iData++) {
        for (jData=0; jData < (hLvlDataDim-1); jData++) {
            /* Get indices for all eight vertices of this sector */
            IdxA = iData*hAndShkDataDim + jData*MedShkDataDim + kData;
            IdxB = IdxA + hAndShkDataDim;
            IdxC = IdxA + MedShkDataDim;
            IdxD = IdxB + MedShkDataDim;
            IdxE = IdxA + 1;
            IdxF = IdxB + 1;
            IdxG = IdxC + 1;
            IdxH = IdxD + 1;
            
            /* Get data for tetrahedron 1 */
            Idx1 = IdxA;
            Idx2 = IdxB;
            Idx3 = IdxC;
            Idx4 = IdxE;
            b1 = bLvlData[Idx1];
            b2 = bLvlData[Idx2];
            b3 = bLvlData[Idx3];
            b4 = bLvlData[Idx4];
            h1 = hLvlData[Idx1];
            h2 = hLvlData[Idx2];
            h3 = hLvlData[Idx3];
            h4 = hLvlData[Idx4];
            Shk1 = MedShkData[Idx1];
            Shk2 = MedShkData[Idx2];
            Shk3 = MedShkData[Idx3];
            Shk4 = MedShkData[Idx4];
            v1 = ValueData[Idx1];
            v2 = ValueData[Idx2];
            v3 = ValueData[Idx3];
            v4 = ValueData[Idx4];
            x1 = xLvlData[Idx1];
            x2 = xLvlData[Idx2];
            x3 = xLvlData[Idx3];
            x4 = xLvlData[Idx4];
            i1 = iLvlData[Idx1];
            i2 = iLvlData[Idx2];
            i3 = iLvlData[Idx3];
            i4 = iLvlData[Idx4];
            dvdh1 = dvdhData[Idx1];
            dvdh2 = dvdhData[Idx2];
            dvdh3 = dvdhData[Idx3];
            dvdh4 = dvdhData[Idx4];
            v_data = (double4)(v1,v2,v3,v4);
            x_data = (double4)(x1,x2,x3,x4);
            i_data = (double4)(i1,i2,i3,i4);
            dvdh_data = (double4)(dvdh1,dvdh2,dvdh3,dvdh4);
            
            /* Find bounding box for tetrahedron 1 */
            bMin = fmin(b1,fmin(b2,fmin(b3,b4)));
            bMax = fmax(b1,fmax(b2,fmax(b3,b4)));
            hMin = fmin(h1,fmin(h2,fmin(h3,h4)));
            hMax = fmax(h1,fmax(h2,fmax(h3,h4)));
            
            /* Find bounding indices for tetrahedron 1 */
            iMin = findIndex(bGridDense,0,bGridDenseSize-1,bMin);
            iMax = findIndex(bGridDense,0,bGridDenseSize-1,bMax);
            jMin = findIndex(hGridDense,0,hGridDenseSize-1,hMin);
            jMax = findIndex(hGridDense,0,hGridDenseSize-1,hMax);
            
            /* Get the transformation matrix for tetrahedron 1 (and the reference point) */
            TransMatrix = make3DtransformationMatrix(b1,h1,Shk1,b2,h2,Shk2,b3,h3,Shk3,b4,h4,Shk4);
            T_row1 = TransMatrix.s0129;
            T_row2 = TransMatrix.s3459;
            T_row3 = TransMatrix.s6789;
            RefPoint = (double4)(b4,h4,Shk4,0.0);
            
            /* Loop over dense grid points */
            for (iDense=iMin; iDense<iMax; iDense++) {
                bLvl = bGridDense[iDense];
                for (jDense=jMin; jDense<jMax; jDense++) {
                    hLvl = hGridDense[jDense];
                    for (kDense=kMin; kDense<kMax; kDense++) {
                        MedShk = ShkGridDense[kDense];
                        
                        /* Get dense point's index and tetrahedral weights */
                        DenseIdx = iDense*hAndShkGridDenseSize + jDense*ShkGridDenseSize + kDense;
                        DensePoint = (double4)(bLvl,hLvl,MedShk,0.0);
                        SectorWeights = calcBarycentricWeights(DensePoint,RefPoint,T_row1,T_row2,T_row3);
                        
                        /* Check whether dense point is "inside" the tetrahedron */
                        InBot = all(SectorWeights >= Low);
                        InTop = all(SectorWeights <= High);
                        if (InBot & InTop) {
                            Value = ValueOut[DenseIdx];
                            vNew = dot(SectorWeights,v_data);
                            if (vNew > Value) {
                                /* If candidate dominates current value, replace with candidate in Out arrays */
                                Value = vNew;
                                xNew = dot(SectorWeights,x_data);
                                iNew = dot(SectorWeights,i_data);
                                dvdhNew = dot(SectorWeights,dvdh_data);
                                ValueOut[DenseIdx] = vNew;
                                xLvlOut[DenseIdx] = xNew;
                                iLvlOut[DenseIdx] = iNew;
                                dvdhOut[DenseIdx] = dvdhNew;
                            }    
                        }
                    }   
                }
            } /* End of dense grid point loop for tetrahedron 1 */
                            
            
            /* Get data for tetrahedron 2 */
            Idx1 = IdxD;
            Idx2 = IdxC;
            Idx3 = IdxB;
            Idx4 = IdxH;
            b1 = bLvlData[Idx1];
            b2 = bLvlData[Idx2];
            b3 = bLvlData[Idx3];
            b4 = bLvlData[Idx4];
            h1 = hLvlData[Idx1];
            h2 = hLvlData[Idx2];
            h3 = hLvlData[Idx3];
            h4 = hLvlData[Idx4];
            Shk1 = MedShkData[Idx1];
            Shk2 = MedShkData[Idx2];
            Shk3 = MedShkData[Idx3];
            Shk4 = MedShkData[Idx4];
            v1 = ValueData[Idx1];
            v2 = ValueData[Idx2];
            v3 = ValueData[Idx3];
            v4 = ValueData[Idx4];
            x1 = xLvlData[Idx1];
            x2 = xLvlData[Idx2];
            x3 = xLvlData[Idx3];
            x4 = xLvlData[Idx4];
            i1 = iLvlData[Idx1];
            i2 = iLvlData[Idx2];
            i3 = iLvlData[Idx3];
            i4 = iLvlData[Idx4];
            dvdh1 = dvdhData[Idx1];
            dvdh2 = dvdhData[Idx2];
            dvdh3 = dvdhData[Idx3];
            dvdh4 = dvdhData[Idx4];
            v_data = (double4)(v1,v2,v3,v4);
            x_data = (double4)(x1,x2,x3,x4);
            i_data = (double4)(i1,i2,i3,i4);
            dvdh_data = (double4)(dvdh1,dvdh2,dvdh3,dvdh4);
            
            /* Find bounding box for tetrahedron 2 */
            bMin = fmin(b1,fmin(b2,fmin(b3,b4)));
            bMax = fmax(b1,fmax(b2,fmax(b3,b4)));
            hMin = fmin(h1,fmin(h2,fmin(h3,h4)));
            hMax = fmax(h1,fmax(h2,fmax(h3,h4)));
            
            /* Find bounding indices for tetrahedron 2 */
            iMin = findIndex(bGridDense,0,bGridDenseSize-1,bMin);
            iMax = findIndex(bGridDense,0,bGridDenseSize-1,bMax);
            jMin = findIndex(hGridDense,0,hGridDenseSize-1,hMin);
            jMax = findIndex(hGridDense,0,hGridDenseSize-1,hMax);
            
            /* Get the transformation matrix for tetrahedron 2 (and the reference point) */
            TransMatrix = make3DtransformationMatrix(b1,h1,Shk1,b2,h2,Shk2,b3,h3,Shk3,b4,h4,Shk4);
            T_row1 = TransMatrix.s0129;
            T_row2 = TransMatrix.s3459;
            T_row3 = TransMatrix.s6789;
            RefPoint = (double4)(b4,h4,Shk4,0.0);
            
            /* Loop over dense grid points */
            for (iDense=iMin; iDense<iMax; iDense++) {
                bLvl = bGridDense[iDense];
                for (jDense=jMin; jDense<jMax; jDense++) {
                    hLvl = hGridDense[jDense];
                    for (kDense=kMin; kDense<kMax; kDense++) {
                        MedShk = ShkGridDense[kDense];
                        
                        /* Get dense point's index and tetrahedral weights */
                        DenseIdx = iDense*hAndShkGridDenseSize + jDense*ShkGridDenseSize + kDense;
                        DensePoint = (double4)(bLvl,hLvl,MedShk,0.0);
                        SectorWeights = calcBarycentricWeights(DensePoint,RefPoint,T_row1,T_row2,T_row3);
                        
                        /* Check whether dense point is "inside" the tetrahedron */
                        InBot = all(SectorWeights >= Low);
                        InTop = all(SectorWeights <= High);
                        if (InBot & InTop) {
                            Value = ValueOut[DenseIdx];
                            vNew = dot(SectorWeights,v_data);
                            if (vNew > Value) {
                                /* If candidate dominates current value, replace with candidate in Out arrays */
                                Value = vNew;
                                xNew = dot(SectorWeights,x_data);
                                iNew = dot(SectorWeights,i_data);
                                dvdhNew = dot(SectorWeights,dvdh_data);
                                ValueOut[DenseIdx] = vNew;
                                xLvlOut[DenseIdx] = xNew;
                                iLvlOut[DenseIdx] = iNew;
                                dvdhOut[DenseIdx] = dvdhNew;
                            }    
                        }
                    }   
                }
            } /* End of dense grid point loop for tetrahedron 2 */
                            
                            
            /* Get data for tetrahedron 3 */
            Idx1 = IdxF;
            Idx2 = IdxE;
            Idx3 = IdxH;
            Idx4 = IdxB;
            b1 = bLvlData[Idx1];
            b2 = bLvlData[Idx2];
            b3 = bLvlData[Idx3];
            b4 = bLvlData[Idx4];
            h1 = hLvlData[Idx1];
            h2 = hLvlData[Idx2];
            h3 = hLvlData[Idx3];
            h4 = hLvlData[Idx4];
            Shk1 = MedShkData[Idx1];
            Shk2 = MedShkData[Idx2];
            Shk3 = MedShkData[Idx3];
            Shk4 = MedShkData[Idx4];
            v1 = ValueData[Idx1];
            v2 = ValueData[Idx2];
            v3 = ValueData[Idx3];
            v4 = ValueData[Idx4];
            x1 = xLvlData[Idx1];
            x2 = xLvlData[Idx2];
            x3 = xLvlData[Idx3];
            x4 = xLvlData[Idx4];
            i1 = iLvlData[Idx1];
            i2 = iLvlData[Idx2];
            i3 = iLvlData[Idx3];
            i4 = iLvlData[Idx4];
            dvdh1 = dvdhData[Idx1];
            dvdh2 = dvdhData[Idx2];
            dvdh3 = dvdhData[Idx3];
            dvdh4 = dvdhData[Idx4];
            v_data = (double4)(v1,v2,v3,v4);
            x_data = (double4)(x1,x2,x3,x4);
            i_data = (double4)(i1,i2,i3,i4);
            dvdh_data = (double4)(dvdh1,dvdh2,dvdh3,dvdh4);
            
            /* Find bounding box for tetrahedron 3 */
            bMin = fmin(b1,fmin(b2,fmin(b3,b4)));
            bMax = fmax(b1,fmax(b2,fmax(b3,b4)));
            hMin = fmin(h1,fmin(h2,fmin(h3,h4)));
            hMax = fmax(h1,fmax(h2,fmax(h3,h4)));
            
            /* Find bounding indices for tetrahedron 3 */
            iMin = findIndex(bGridDense,0,bGridDenseSize-1,bMin);
            iMax = findIndex(bGridDense,0,bGridDenseSize-1,bMax);
            jMin = findIndex(hGridDense,0,hGridDenseSize-1,hMin);
            jMax = findIndex(hGridDense,0,hGridDenseSize-1,hMax);
            
            /* Get the transformation matrix for tetrahedron 3 (and the reference point) */
            TransMatrix = make3DtransformationMatrix(b1,h1,Shk1,b2,h2,Shk2,b3,h3,Shk3,b4,h4,Shk4);
            T_row1 = TransMatrix.s0129;
            T_row2 = TransMatrix.s3459;
            T_row3 = TransMatrix.s6789;
            RefPoint = (double4)(b4,h4,Shk4,0.0);
            
            /* Loop over dense grid points */
            for (iDense=iMin; iDense<iMax; iDense++) {
                bLvl = bGridDense[iDense];
                for (jDense=jMin; jDense<jMax; jDense++) {
                    hLvl = hGridDense[jDense];
                    for (kDense=kMin; kDense<kMax; kDense++) {
                        MedShk = ShkGridDense[kDense];
                        
                        /* Get dense point's index and tetrahedral weights */
                        DenseIdx = iDense*hAndShkGridDenseSize + jDense*ShkGridDenseSize + kDense;
                        DensePoint = (double4)(bLvl,hLvl,MedShk,0.0);
                        SectorWeights = calcBarycentricWeights(DensePoint,RefPoint,T_row1,T_row2,T_row3);
                        
                        /* Check whether dense point is "inside" the tetrahedron */
                        InBot = all(SectorWeights >= Low);
                        InTop = all(SectorWeights <= High);
                        if (InBot & InTop) {
                            Value = ValueOut[DenseIdx];
                            vNew = dot(SectorWeights,v_data);
                            if (vNew > Value) {
                                /* If candidate dominates current value, replace with candidate in Out arrays */
                                Value = vNew;
                                xNew = dot(SectorWeights,x_data);
                                iNew = dot(SectorWeights,i_data);
                                dvdhNew = dot(SectorWeights,dvdh_data);
                                ValueOut[DenseIdx] = vNew;
                                xLvlOut[DenseIdx] = xNew;
                                iLvlOut[DenseIdx] = iNew;
                                dvdhOut[DenseIdx] = dvdhNew;
                            }    
                        }
                    }   
                }
            } /* End of dense grid point loop for tetrahedron 3 */
                            
                            
            /* Get data for tetrahedron 4 */
            Idx1 = IdxG;
            Idx2 = IdxH;
            Idx3 = IdxE;
            Idx4 = IdxC;
            b1 = bLvlData[Idx1];
            b2 = bLvlData[Idx2];
            b3 = bLvlData[Idx3];
            b4 = bLvlData[Idx4];
            h1 = hLvlData[Idx1];
            h2 = hLvlData[Idx2];
            h3 = hLvlData[Idx3];
            h4 = hLvlData[Idx4];
            Shk1 = MedShkData[Idx1];
            Shk2 = MedShkData[Idx2];
            Shk3 = MedShkData[Idx3];
            Shk4 = MedShkData[Idx4];
            v1 = ValueData[Idx1];
            v2 = ValueData[Idx2];
            v3 = ValueData[Idx3];
            v4 = ValueData[Idx4];
            x1 = xLvlData[Idx1];
            x2 = xLvlData[Idx2];
            x3 = xLvlData[Idx3];
            x4 = xLvlData[Idx4];
            i1 = iLvlData[Idx1];
            i2 = iLvlData[Idx2];
            i3 = iLvlData[Idx3];
            i4 = iLvlData[Idx4];
            dvdh1 = dvdhData[Idx1];
            dvdh2 = dvdhData[Idx2];
            dvdh3 = dvdhData[Idx3];
            dvdh4 = dvdhData[Idx4];
            v_data = (double4)(v1,v2,v3,v4);
            x_data = (double4)(x1,x2,x3,x4);
            i_data = (double4)(i1,i2,i3,i4);
            dvdh_data = (double4)(dvdh1,dvdh2,dvdh3,dvdh4);
            
            /* Find bounding box for tetrahedron 4 */
            bMin = fmin(b1,fmin(b2,fmin(b3,b4)));
            bMax = fmax(b1,fmax(b2,fmax(b3,b4)));
            hMin = fmin(h1,fmin(h2,fmin(h3,h4)));
            hMax = fmax(h1,fmax(h2,fmax(h3,h4)));
            
            /* Find bounding indices for tetrahedron 4 */
            iMin = findIndex(bGridDense,0,bGridDenseSize-1,bMin);
            iMax = findIndex(bGridDense,0,bGridDenseSize-1,bMax);
            jMin = findIndex(hGridDense,0,hGridDenseSize-1,hMin);
            jMax = findIndex(hGridDense,0,hGridDenseSize-1,hMax);
            
            /* Get the transformation matrix for tetrahedron 4 (and the reference point) */
            TransMatrix = make3DtransformationMatrix(b1,h1,Shk1,b2,h2,Shk2,b3,h3,Shk3,b4,h4,Shk4);
            T_row1 = TransMatrix.s0129;
            T_row2 = TransMatrix.s3459;
            T_row3 = TransMatrix.s6789;
            RefPoint = (double4)(b4,h4,Shk4,0.0);
            
            /* Loop over dense grid points */
            for (iDense=iMin; iDense<iMax; iDense++) {
                bLvl = bGridDense[iDense];
                for (jDense=jMin; jDense<jMax; jDense++) {
                    hLvl = hGridDense[jDense];
                    for (kDense=kMin; kDense<kMax; kDense++) {
                        MedShk = ShkGridDense[kDense];
                        
                        /* Get dense point's index and tetrahedral weights */
                        DenseIdx = iDense*hAndShkGridDenseSize + jDense*ShkGridDenseSize + kDense;
                        DensePoint = (double4)(bLvl,hLvl,MedShk,0.0);
                        SectorWeights = calcBarycentricWeights(DensePoint,RefPoint,T_row1,T_row2,T_row3);
                        
                        /* Check whether dense point is "inside" the tetrahedron */
                        InBot = all(SectorWeights >= Low);
                        InTop = all(SectorWeights <= High);
                        if (InBot & InTop) {
                            Value = ValueOut[DenseIdx];
                            vNew = dot(SectorWeights,v_data);
                            if (vNew > Value) {
                                /* If candidate dominates current value, replace with candidate in Out arrays */
                                Value = vNew;
                                xNew = dot(SectorWeights,x_data);
                                iNew = dot(SectorWeights,i_data);
                                dvdhNew = dot(SectorWeights,dvdh_data);
                                ValueOut[DenseIdx] = vNew;
                                xLvlOut[DenseIdx] = xNew;
                                iLvlOut[DenseIdx] = iNew;
                                dvdhOut[DenseIdx] = dvdhNew;
                            }    
                        }
                    }   
                }
            } /* End of dense grid point loop for tetrahedron 4 */
                            
                            
            /* Get data for tetrahedron 5 */
            Idx1 = IdxH;
            Idx2 = IdxB;
            Idx3 = IdxC;
            Idx4 = IdxE;
            b1 = bLvlData[Idx1];
            b2 = bLvlData[Idx2];
            b3 = bLvlData[Idx3];
            b4 = bLvlData[Idx4];
            h1 = hLvlData[Idx1];
            h2 = hLvlData[Idx2];
            h3 = hLvlData[Idx3];
            h4 = hLvlData[Idx4];
            Shk1 = MedShkData[Idx1];
            Shk2 = MedShkData[Idx2];
            Shk3 = MedShkData[Idx3];
            Shk4 = MedShkData[Idx4];
            v1 = ValueData[Idx1];
            v2 = ValueData[Idx2];
            v3 = ValueData[Idx3];
            v4 = ValueData[Idx4];
            x1 = xLvlData[Idx1];
            x2 = xLvlData[Idx2];
            x3 = xLvlData[Idx3];
            x4 = xLvlData[Idx4];
            i1 = iLvlData[Idx1];
            i2 = iLvlData[Idx2];
            i3 = iLvlData[Idx3];
            i4 = iLvlData[Idx4];
            dvdh1 = dvdhData[Idx1];
            dvdh2 = dvdhData[Idx2];
            dvdh3 = dvdhData[Idx3];
            dvdh4 = dvdhData[Idx4];
            v_data = (double4)(v1,v2,v3,v4);
            x_data = (double4)(x1,x2,x3,x4);
            i_data = (double4)(i1,i2,i3,i4);
            dvdh_data = (double4)(dvdh1,dvdh2,dvdh3,dvdh4);
            
            /* Find bounding box for tetrahedron 5 */
            bMin = fmin(b1,fmin(b2,fmin(b3,b4)));
            bMax = fmax(b1,fmax(b2,fmax(b3,b4)));
            hMin = fmin(h1,fmin(h2,fmin(h3,h4)));
            hMax = fmax(h1,fmax(h2,fmax(h3,h4)));
            
            /* Find bounding indices for tetrahedron 5 */
            iMin = findIndex(bGridDense,0,bGridDenseSize-1,bMin);
            iMax = findIndex(bGridDense,0,bGridDenseSize-1,bMax);
            jMin = findIndex(hGridDense,0,hGridDenseSize-1,hMin);
            jMax = findIndex(hGridDense,0,hGridDenseSize-1,hMax);
            
            /* Get the transformation matrix for tetrahedron 5 (and the reference point) */
            TransMatrix = make3DtransformationMatrix(b1,h1,Shk1,b2,h2,Shk2,b3,h3,Shk3,b4,h4,Shk4);
            T_row1 = TransMatrix.s0129;
            T_row2 = TransMatrix.s3459;
            T_row3 = TransMatrix.s6789;
            RefPoint = (double4)(b4,h4,Shk4,0.0);
            
            /* Loop over dense grid points */
            for (iDense=iMin; iDense<iMax; iDense++) {
                bLvl = bGridDense[iDense];
                for (jDense=jMin; jDense<jMax; jDense++) {
                    hLvl = hGridDense[jDense];
                    for (kDense=kMin; kDense<kMax; kDense++) {
                        MedShk = ShkGridDense[kDense];
                        
                        /* Get dense point's index and tetrahedral weights */
                        DenseIdx = iDense*hAndShkGridDenseSize + jDense*ShkGridDenseSize + kDense;
                        DensePoint = (double4)(bLvl,hLvl,MedShk,0.0);
                        SectorWeights = calcBarycentricWeights(DensePoint,RefPoint,T_row1,T_row2,T_row3);
                        
                        /* Check whether dense point is "inside" the tetrahedron */
                        InBot = all(SectorWeights >= Low);
                        InTop = all(SectorWeights <= High);
                        if (InBot & InTop) {
                            Value = ValueOut[DenseIdx];
                            vNew = dot(SectorWeights,v_data);
                            if (vNew > Value) {
                                /* If candidate dominates current value, replace with candidate in Out arrays */
                                Value = vNew;
                                xNew = dot(SectorWeights,x_data);
                                iNew = dot(SectorWeights,i_data);
                                dvdhNew = dot(SectorWeights,dvdh_data);
                                ValueOut[DenseIdx] = vNew;
                                xLvlOut[DenseIdx] = xNew;
                                iLvlOut[DenseIdx] = iNew;
                                dvdhOut[DenseIdx] = dvdhNew;
                            }    
                        }
                    }   
                }
            } /* End of dense grid point loop for tetrahedron 5 */
        }
    }
}