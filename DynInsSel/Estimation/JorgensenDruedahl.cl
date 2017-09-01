# pragma OPENCL EXTENSION cl_khr_fp64 : enable

/* Barycentric weight calculator */
inline double3 calcBarycentricWeights(
     double Ax
    ,double Ay
    ,double Bx
    ,double By
    ,double Cx
    ,double Cy
    ,double X
    ,double Y
    ) {
    double denom = (By-Cy)*(Ax-Cx) + (Cx-Bx)*(Ay-Cy);
    double Aweight = ((By-Cy)*(X-Cx) + (Cx-Bx)*(Y-Cy))/denom;
    double Bweight = ((Cy-Ay)*(X-Cx) + (Ax-Cx)*(Y-Cy))/denom;
    double Cweight = 1.0 - Aweight - Bweight;
    double3 weights = (double3)(Aweight,Bweight,Cweight);
    return weights;
}


/* Kernel for implementing Jorgensen-Druedahl in the DynInsSel model */
__kernel void doJorgensenDruedahlFix(
     __global double *mLvlData           /* data on endogenous mLvl */
    ,__global double *MedShkData         /* data on medical shocks  */
    ,__global double *ValueData          /* data on value at (m,Shk) */
    ,__global double *xLvlData           /* data on optimal xLvl    */
    ,__global double *mGridDense         /* exogenous grid of mLvl  */
    ,__global double *ShkGridDense       /* exogenous grid of MedShk */
    ,__global double *xLvlOut            /* J-D fixed xLvl to return */
    ,__global double *ValueOut           /* J-D fixed value to return */
    ,__global int *IntegerInputs         /* integers that characterize problem */
    ) {

    /* Initialize some variables for use in the main loop */
    double mA;
    double mB;
    double mC;
    double ShkA;
    double ShkB;
    double ShkC;
    double vA;
    double vB;
    double vC;
    double xA;
    double xB;
    double xC;
    double mMin;
    double mMax;
    double ShkMin;
    double ShkMax;
    double vNew = 0.0;
    double xNew;
    double3 SectorWeights = (double3)(0.0,0.0,0.0);
    int IdxA;
    int IdxB;
    int IdxC;

    /* Unpack the integer inputs */
    int mLvlDataDim = IntegerInputs[0];
    int MedShkDataDim = IntegerInputs[1];
    int mGridDenseSize = IntegerInputs[2];
    int ShkGridDenseSize = IntegerInputs[3];
    int ThreadCount = IntegerInputs[4];

    
    /* Initialize this thread's id and get this thread's constant (mLvl,MedShk) identity */
    int Gid = get_global_id(0);     /* global thread id */
    if (Gid >= ThreadCount) {
        return;
    }
    int mGridIdx = Gid/ShkGridDenseSize;
    int ShkGridIdx = Gid - mGridIdx*ShkGridDenseSize;
    double mLvl = mGridDense[mGridIdx];
    double MedShk = ShkGridDense[ShkGridIdx];

    /* Initialize xLvl and value for output */
    double xLvl = xLvlOut[Gid];
    double Value = ValueOut[Gid];

    /* Loop over each triangular sector of (mLvl,MedShk) from the data */
    int i = 0;
    int j = 0;
    double Low = -0.01;
    double High = 1.01;
    while (i < (mLvlDataDim-1)) {
        j = 0;
        while (j < (MedShkDataDim-1)) {
            /* Get location data for lower triangle */
            IdxA = i*MedShkDataDim + j;
            IdxB = IdxA + MedShkDataDim;
            IdxC = IdxB + 1;
            mA = mLvlData[IdxA];
            mB = mLvlData[IdxB];
            mC = mLvlData[IdxC];
            ShkA = MedShkData[IdxA];
            ShkB = MedShkData[IdxB];
            ShkC = MedShkData[IdxC];

            /* Find bounding box for lower triangle */
            mMin = fmin(fmin(mA,mB),mC);
            mMax = fmax(fmax(mA,mB),mC);
            ShkMin = fmin(fmin(ShkA,ShkB),ShkC);
            ShkMax = fmax(fmax(ShkA,ShkB),ShkC);

            /* If self is inside bounding box, calc barycentric weights */
            if ((mLvl >= mMin) & (mLvl <= mMax) & (MedShk >= ShkMin) & (MedShk <= ShkMax)) {
                SectorWeights = calcBarycentricWeights(mA,ShkA,mB,ShkB,mC,ShkC,mLvl,MedShk);

                /* If barycentric weights all between 0 and 1, evaluate vNew */
                if ((SectorWeights.x >= Low) & (SectorWeights.y >= Low) & (SectorWeights.z >= Low) & (SectorWeights.x <= High) & (SectorWeights.y <= High) & (SectorWeights.z <= High)) {
                    vA = ValueData[IdxA];
                    vB = ValueData[IdxB];
                    vC = ValueData[IdxC];
                    vNew = SectorWeights.x*vA + SectorWeights.y*vB + SectorWeights.z*vC;

                    /* If vNew is better than current v, replace v and xLvl in Out */
                    if (vNew > Value) {
                        xA = xLvlData[IdxA];
                        xB = xLvlData[IdxB];
                        xC = xLvlData[IdxC];
                        xNew = SectorWeights.x*xA + SectorWeights.y*xB + SectorWeights.z*xC;
                        xLvl = xNew;
                        Value = vNew;
                    }
                }
            } /* End of checking lower triangle */

            /* Get location data for upper triangle (only need to change B) */
            IdxB = IdxA + 1;
            mB = mLvlData[IdxB];
            ShkB = MedShkData[IdxB];
            
            /* Find bounding box for upper triangle (Shk bounds don't change) */
            mMin = fmin(fmin(mA,mB),mC);
            mMax = fmax(fmax(mA,mB),mC);

            /* If self is inside bounding box, calc barycentric weights */
            if ((mLvl >= mMin) & (mLvl <= mMax) & (MedShk >= ShkMin) & (MedShk <= ShkMax)) {
                SectorWeights = calcBarycentricWeights(mA,ShkA,mB,ShkB,mC,ShkC,mLvl,MedShk);

                /* If barycentric weights all between 0 and 1, evaluate vNew */
                if ((SectorWeights.x >= Low) & (SectorWeights.y >= Low) & (SectorWeights.z >= Low) & (SectorWeights.x <= High) & (SectorWeights.y <= High) & (SectorWeights.z <= High)) {
                    vA = ValueData[IdxA];
                    vB = ValueData[IdxB];
                    vC = ValueData[IdxC];
                    vNew = SectorWeights.x*vA + SectorWeights.y*vB + SectorWeights.z*vC;

                    /* If vNew is better than current v, replace v and xLvl in Out */
                    if (vNew > Value) {
                        xA = xLvlData[IdxA];
                        xB = xLvlData[IdxB];
                        xC = xLvlData[IdxC];
                        xNew = SectorWeights.x*xA + SectorWeights.y*xB + SectorWeights.z*xC;
                        xLvl = xNew;
                        Value = vNew;
                    }
                }
            } /* End of checking upper triangle */

            j++; /* Move to next MedShk for this mLvl in the data */
         }
         i++; /* Move to next mLvl in the data, resetting MedShk */
    }

    /* Store the final xLvl and value in the Out buffers */
    xLvlOut[Gid] = xLvl;
    ValueOut[Gid] = Value;
}
