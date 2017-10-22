# pragma OPENCL EXTENSION cl_khr_fp64 : enable

/* Barycentric weight calculator */
inline double4 calcBarycentricWeights(
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
    ,double X
    ,double Y
    ,double Z
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
    double Qx = X - Dx;
    double Qy = Y - Dy;
    double Qz = Z - Dz;
    double Aweight = b11*Qx + b12*Qy + b13*Qz;
    double Bweight = b21*Qx + b22*Qy + b23*Qz;
    double Cweight = b31*Qx + b32*Qy + b33*Qz;
    double Dweight = 1.0 - Aweight - Bweight - Cweight;
    double4 weights = (double4)(Aweight,Bweight,Cweight,Dweight);
    return weights;
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

    /* Initialize some variables for use in the main loop */
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
    double ShkMin;
    double ShkMax;
    double vNew = 0.0;
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

    /* Unpack the integer inputs */
    int bLvlDataDim = IntegerInputs[0];
    int hLvlDataDim = IntegerInputs[1];
    int MedShkDataDim = IntegerInputs[2];
    int bGridDenseSize = IntegerInputs[3];
    int hGridDenseSize = IntegerInputs[4];
    int ShkGridDenseSize = IntegerInputs[5];
    int ThreadCount = IntegerInputs[6];
    
    /* Initialize this thread's id and get this thread's constant (mLvl,MedShk) identity */
    int Gid = get_global_id(0);     /* global thread id */
    if (Gid >= ThreadCount) {
        return;
    }
    int bGridIdx = Gid/(ShkGridDenseSize*hGridDenseSize);
    int hGridIdx = (Gid - bGridIdx*ShkGridDenseSize*hGridDenseSize)/ShkGridDenseSize;
    int ShkGridIdx = Gid - bGridIdx*ShkGridDenseSize*hGridDenseSize - hGridIdx*ShkGridDenseSize;
    double bLvl = bGridDense[bGridIdx];
    double hLvl = hGridDense[hGridIdx];
    double MedShk = ShkGridDense[ShkGridIdx];

    /* Initialize xLvl,iLvl, Value, and dvdh for output */
    double xLvl = xLvlOut[Gid];
    double iLvl = iLvlOut[Gid];
    double Value = ValueOut[Gid];
    double dvdh = dvdhOut[Gid];
    int hAndShkDataDim = hLvlDataDim*MedShkDataDim;
    
    /* Loop over each triangular sector of (mLvl,MedShk) from the data */
    int i = 0;
    int j = 0;
    int k = 0;
    double Low = -0.10;
    double High = 1.10;
    while (k < (MedShkDataDim-1)) {
        if ((MedShk >= MedShkData[k]) & (MedShk <= MedShkData[k+1])) { /* Only look at MedShk layers that are relevant */
            j = 0;
            while (j < (hLvlDataDim-1)) {
                i = 0;
                while (i < (bLvlDataDim-1)) {
                    /* Get indices for all eight vertices of this sector */
                    IdxA = i*hAndShkDataDim + j*MedShkDataDim + k;
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
                    
                    /* Find bounding box for tetrahedron 1 */
                    bMin = fmin(b1,fmin(b2,fmin(b3,b4)));
                    bMax = fmax(b1,fmax(b2,fmax(b3,b4)));
                    hMin = fmin(h1,fmin(h2,fmin(h3,h4)));
                    hMax = fmax(h1,fmax(h2,fmax(h3,h4)));
                    
                    /* If self is inside bounding box, calc barycentric weights */
                    if ((bLvl >= bMin) & (bLvl <= bMax) & (hLvl >= hMin) & (hLvl <= hMax)) {
                        SectorWeights = calcBarycentricWeights(b1,h1,Shk1,b2,h2,Shk2,b3,h3,Shk3,b4,h4,Shk4,bLvl,hLvl,MedShk);
        
                        /* If barycentric weights all between 0 and 1, evaluate vNew */
                        if ((SectorWeights.x >= Low) & (SectorWeights.y >= Low) & (SectorWeights.z >= Low) & (SectorWeights.w >= Low) & (SectorWeights.x <= High) & (SectorWeights.y <= High) & (SectorWeights.z <= High) & (SectorWeights.w <= High)) {
                            v1 = ValueData[Idx1];
                            v2 = ValueData[Idx2];
                            v3 = ValueData[Idx3];
                            v4 = ValueData[Idx4];
                            vNew = SectorWeights.x*v1 + SectorWeights.y*v2 + SectorWeights.z*v3 + SectorWeights.w*v4;
                            
                            /* If vNew is better than current v, replace v and xLvl in Out */
                            if (vNew > Value) {
                                x1 = xLvlData[Idx1];
                                x2 = xLvlData[Idx2];
                                x3 = xLvlData[Idx3];
                                x4 = xLvlData[Idx4];
                                xNew = SectorWeights.x*x1 + SectorWeights.y*x2 + SectorWeights.z*x3 + SectorWeights.w*x4;
                                i1 = iLvlData[Idx1];
                                i2 = iLvlData[Idx2];
                                i3 = iLvlData[Idx3];
                                i4 = iLvlData[Idx4];
                                iNew = SectorWeights.x*i1 + SectorWeights.y*i2 + SectorWeights.z*i3 + SectorWeights.w*i4;
                                dvdh1 = dvdhData[Idx1];
                                dvdh2 = dvdhData[Idx2];
                                dvdh3 = dvdhData[Idx3];
                                dvdh4 = dvdhData[Idx4];
                                dvdhNew = SectorWeights.x*dvdh1 + SectorWeights.y*dvdh2 + SectorWeights.z*dvdh3 + SectorWeights.w*dvdh4;
                                xLvl = xNew;
                                iLvl = iNew;
                                dvdh = dvdhNew;
                                Value = vNew;
                            }
                            
                        }
                    } /* End of checking tetrahedron 1 */
                            
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
                    
                    /* Find bounding box for tetrahedron 2 */
                    bMin = fmin(b1,fmin(b2,fmin(b3,b4)));
                    bMax = fmax(b1,fmax(b2,fmax(b3,b4)));
                    hMin = fmin(h1,fmin(h2,fmin(h3,h4)));
                    hMax = fmax(h1,fmax(h2,fmax(h3,h4)));
                    
                    /* If self is inside bounding box, calc barycentric weights */
                    if ((bLvl >= bMin) & (bLvl <= bMax) & (hLvl >= hMin) & (hLvl <= hMax)) {
                        SectorWeights = calcBarycentricWeights(b1,h1,Shk1,b2,h2,Shk2,b3,h3,Shk3,b4,h4,Shk4,bLvl,hLvl,MedShk);
        
                        /* If barycentric weights all between 0 and 1, evaluate vNew */
                        if ((SectorWeights.x >= Low) & (SectorWeights.y >= Low) & (SectorWeights.z >= Low) & (SectorWeights.w >= Low) & (SectorWeights.x <= High) & (SectorWeights.y <= High) & (SectorWeights.z <= High) & (SectorWeights.w <= High)) {
                            v1 = ValueData[Idx1];
                            v2 = ValueData[Idx2];
                            v3 = ValueData[Idx3];
                            v4 = ValueData[Idx4];
                            vNew = SectorWeights.x*v1 + SectorWeights.y*v2 + SectorWeights.z*v3 + SectorWeights.w*v4;
                            
                            /* If vNew is better than current v, replace v and xLvl in Out */
                            if (vNew > Value) {
                                x1 = xLvlData[Idx1];
                                x2 = xLvlData[Idx2];
                                x3 = xLvlData[Idx3];
                                x4 = xLvlData[Idx4];
                                xNew = SectorWeights.x*x1 + SectorWeights.y*x2 + SectorWeights.z*x3 + SectorWeights.w*x4;
                                i1 = iLvlData[Idx1];
                                i2 = iLvlData[Idx2];
                                i3 = iLvlData[Idx3];
                                i4 = iLvlData[Idx4];
                                iNew = SectorWeights.x*i1 + SectorWeights.y*i2 + SectorWeights.z*i3 + SectorWeights.w*i4;
                                dvdh1 = dvdhData[Idx1];
                                dvdh2 = dvdhData[Idx2];
                                dvdh3 = dvdhData[Idx3];
                                dvdh4 = dvdhData[Idx4];
                                dvdhNew = SectorWeights.x*dvdh1 + SectorWeights.y*dvdh2 + SectorWeights.z*dvdh3 + SectorWeights.w*dvdh4;
                                xLvl = xNew;
                                iLvl = iNew;
                                dvdh = dvdhNew;
                                Value = vNew;
                            }
                        }
                    } /* End of checking tetrahedron 2 */
                            
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
                    
                    /* Find bounding box for tetrahedron 3 */
                    bMin = fmin(b1,fmin(b2,fmin(b3,b4)));
                    bMax = fmax(b1,fmax(b2,fmax(b3,b4)));
                    hMin = fmin(h1,fmin(h2,fmin(h3,h4)));
                    hMax = fmax(h1,fmax(h2,fmax(h3,h4)));
                    
                    /* If self is inside bounding box, calc barycentric weights */
                    if ((bLvl >= bMin) & (bLvl <= bMax) & (hLvl >= hMin) & (hLvl <= hMax)) {
                        SectorWeights = calcBarycentricWeights(b1,h1,Shk1,b2,h2,Shk2,b3,h3,Shk3,b4,h4,Shk4,bLvl,hLvl,MedShk);
        
                        /* If barycentric weights all between 0 and 1, evaluate vNew */
                        if ((SectorWeights.x >= Low) & (SectorWeights.y >= Low) & (SectorWeights.z >= Low) & (SectorWeights.w >= Low) & (SectorWeights.x <= High) & (SectorWeights.y <= High) & (SectorWeights.z <= High) & (SectorWeights.w <= High)) {
                            v1 = ValueData[Idx1];
                            v2 = ValueData[Idx2];
                            v3 = ValueData[Idx3];
                            v4 = ValueData[Idx4];
                            vNew = SectorWeights.x*v1 + SectorWeights.y*v2 + SectorWeights.z*v3 + SectorWeights.w*v4;
                            
                            /* If vNew is better than current v, replace v and xLvl in Out */
                            if (vNew > Value) {
                                x1 = xLvlData[Idx1];
                                x2 = xLvlData[Idx2];
                                x3 = xLvlData[Idx3];
                                x4 = xLvlData[Idx4];
                                xNew = SectorWeights.x*x1 + SectorWeights.y*x2 + SectorWeights.z*x3 + SectorWeights.w*x4;
                                i1 = iLvlData[Idx1];
                                i2 = iLvlData[Idx2];
                                i3 = iLvlData[Idx3];
                                i4 = iLvlData[Idx4];
                                iNew = SectorWeights.x*i1 + SectorWeights.y*i2 + SectorWeights.z*i3 + SectorWeights.w*i4;
                                dvdh1 = dvdhData[Idx1];
                                dvdh2 = dvdhData[Idx2];
                                dvdh3 = dvdhData[Idx3];
                                dvdh4 = dvdhData[Idx4];
                                dvdhNew = SectorWeights.x*dvdh1 + SectorWeights.y*dvdh2 + SectorWeights.z*dvdh3 + SectorWeights.w*dvdh4;
                                xLvl = xNew;
                                iLvl = iNew;
                                dvdh = dvdhNew;
                                Value = vNew;
                            }
                        }
                    } /* End of checking tetrahedron 3 */
                            
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
                    
                    /* Find bounding box for tetrahedron 4 */
                    bMin = fmin(b1,fmin(b2,fmin(b3,b4)));
                    bMax = fmax(b1,fmax(b2,fmax(b3,b4)));
                    hMin = fmin(h1,fmin(h2,fmin(h3,h4)));
                    hMax = fmax(h1,fmax(h2,fmax(h3,h4)));
                    
                    /* If self is inside bounding box, calc barycentric weights */
                    if ((bLvl >= bMin) & (bLvl <= bMax) & (hLvl >= hMin) & (hLvl <= hMax)) {
                        SectorWeights = calcBarycentricWeights(b1,h1,Shk1,b2,h2,Shk2,b3,h3,Shk3,b4,h4,Shk4,bLvl,hLvl,MedShk);
        
                        /* If barycentric weights all between 0 and 1, evaluate vNew */
                        if ((SectorWeights.x >= Low) & (SectorWeights.y >= Low) & (SectorWeights.z >= Low) & (SectorWeights.w >= Low) & (SectorWeights.x <= High) & (SectorWeights.y <= High) & (SectorWeights.z <= High) & (SectorWeights.w <= High)) {
                            v1 = ValueData[Idx1];
                            v2 = ValueData[Idx2];
                            v3 = ValueData[Idx3];
                            v4 = ValueData[Idx4];
                            vNew = SectorWeights.x*v1 + SectorWeights.y*v2 + SectorWeights.z*v3 + SectorWeights.w*v4;
                            
                            /* If vNew is better than current v, replace v and xLvl in Out */
                            if (vNew > Value) {
                                x1 = xLvlData[Idx1];
                                x2 = xLvlData[Idx2];
                                x3 = xLvlData[Idx3];
                                x4 = xLvlData[Idx4];
                                xNew = SectorWeights.x*x1 + SectorWeights.y*x2 + SectorWeights.z*x3 + SectorWeights.w*x4;
                                i1 = iLvlData[Idx1];
                                i2 = iLvlData[Idx2];
                                i3 = iLvlData[Idx3];
                                i4 = iLvlData[Idx4];
                                iNew = SectorWeights.x*i1 + SectorWeights.y*i2 + SectorWeights.z*i3 + SectorWeights.w*i4;
                                dvdh1 = dvdhData[Idx1];
                                dvdh2 = dvdhData[Idx2];
                                dvdh3 = dvdhData[Idx3];
                                dvdh4 = dvdhData[Idx4];
                                dvdhNew = SectorWeights.x*dvdh1 + SectorWeights.y*dvdh2 + SectorWeights.z*dvdh3 + SectorWeights.w*dvdh4;
                                xLvl = xNew;
                                iLvl = iNew;
                                dvdh = dvdhNew;
                                Value = vNew;
                            }
                        }
                    } /* End of checking tetrahedron 4 */
                            
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
                    
                    /* Find bounding box for tetrahedron 5 */
                    bMin = fmin(b1,fmin(b2,fmin(b3,b4)));
                    bMax = fmax(b1,fmax(b2,fmax(b3,b4)));
                    hMin = fmin(h1,fmin(h2,fmin(h3,h4)));
                    hMax = fmax(h1,fmax(h2,fmax(h3,h4)));
                    
                    /* If self is inside bounding box, calc barycentric weights */
                    if ((bLvl >= bMin) & (bLvl <= bMax) & (hLvl >= hMin) & (hLvl <= hMax)) {
                        SectorWeights = calcBarycentricWeights(b1,h1,Shk1,b2,h2,Shk2,b3,h3,Shk3,b4,h4,Shk4,bLvl,hLvl,MedShk);
        
                        /* If barycentric weights all between 0 and 1, evaluate vNew */
                        if ((SectorWeights.x >= Low) & (SectorWeights.y >= Low) & (SectorWeights.z >= Low) & (SectorWeights.w >= Low) & (SectorWeights.x <= High) & (SectorWeights.y <= High) & (SectorWeights.z <= High) & (SectorWeights.w <= High)) {
                            v1 = ValueData[Idx1];
                            v2 = ValueData[Idx2];
                            v3 = ValueData[Idx3];
                            v4 = ValueData[Idx4];
                            vNew = SectorWeights.x*v1 + SectorWeights.y*v2 + SectorWeights.z*v3 + SectorWeights.w*v4;
                            
                            /* If vNew is better than current v, replace v and xLvl in Out */
                            if (vNew > Value) {
                                x1 = xLvlData[Idx1];
                                x2 = xLvlData[Idx2];
                                x3 = xLvlData[Idx3];
                                x4 = xLvlData[Idx4];
                                xNew = SectorWeights.x*x1 + SectorWeights.y*x2 + SectorWeights.z*x3 + SectorWeights.w*x4;
                                i1 = iLvlData[Idx1];
                                i2 = iLvlData[Idx2];
                                i3 = iLvlData[Idx3];
                                i4 = iLvlData[Idx4];
                                iNew = SectorWeights.x*i1 + SectorWeights.y*i2 + SectorWeights.z*i3 + SectorWeights.w*i4;
                                dvdh1 = dvdhData[Idx1];
                                dvdh2 = dvdhData[Idx2];
                                dvdh3 = dvdhData[Idx3];
                                dvdh4 = dvdhData[Idx4];
                                dvdhNew = SectorWeights.x*dvdh1 + SectorWeights.y*dvdh2 + SectorWeights.z*dvdh3 + SectorWeights.w*dvdh4;
                                xLvl = xNew;
                                iLvl = iNew;
                                dvdh = dvdhNew;
                                Value = vNew;
                            }
                        }
                    } /* End of checking tetrahedron 5 */
                      
                    i++;
                }
                j++;
            }
        }
        k++;
    }
                
    /* Store the final results in the Out buffers */
    xLvlOut[Gid] = xLvl;
    iLvlOut[Gid] = iLvl;
    ValueOut[Gid] = Value;
    dvdhOut[Gid] = dvdh;
}
              
