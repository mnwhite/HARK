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
    ,__global double *DevData            /* data on medical shock deviations */
    ,__global double *ValueData          /* data on value at (m,Dev) */
    ,__global double *xLvlData           /* data on optimal xLvl    */
    ,__global double *mGridDense         /* exogenous grid of mLvl  */
    ,__global double *DevGridDense       /* exogenous grid of Dev */
    ,__global double *xLvlOut            /* J-D fixed xLvl to return */
    ,__global double *ValueOut           /* J-D fixed value to return */
    ,__global int *IntegerInputs         /* integers that characterize problem */
    ) {

    /* Initialize some variables for use in the main loop */
    double mA;
    double mB;
    double mC;
    double DevA;
    double DevB;
    double DevC;
    double vA;
    double vB;
    double vC;
    double xA;
    double xB;
    double xC;
    double mMin;
    double mMax;
    double DevMin;
    double DevMax;
    double vNew = 0.0;
    double xNew;
    double3 SectorWeights = (double3)(0.0,0.0,0.0);
    int IdxA;
    int IdxB;
    int IdxC;

    /* Unpack the integer inputs */
    int mLvlDataDim = IntegerInputs[0];
    int DevDataDim = IntegerInputs[1];
    int mGridDenseSize = IntegerInputs[2];
    int DevGridDenseSize = IntegerInputs[3];
    int ThreadCount = IntegerInputs[4];

    /* Initialize this thread's id and get this thread's constant (mLvl,Dev) identity */
    int Gid = get_global_id(0);     /* global thread id */
    if (Gid >= ThreadCount) {
        return;
    }
    int mGridIdx = Gid/DevGridDenseSize;
    int DevGridIdx = Gid - mGridIdx*DevGridDenseSize;
    double mLvl = mGridDense[mGridIdx];
    double Dev = DevGridDense[DevGridIdx];

    /* Initialize xLvl and value for output */
    double xLvl = xLvlOut[Gid];
    double Value = ValueOut[Gid];

    /* Loop over each triangular sector of (mLvl,Dev) from the data */
    int i = 0;
    int j = 0;
    double Low = -0.001;
    double High = 1.001;
    while (i < (mLvlDataDim-1)) {
        j = 0;
        while (j < (DevDataDim-1)) {
            /* Get location data for lower triangle */
            IdxA = i*DevDataDim + j;
            IdxB = IdxA + DevDataDim;
            IdxC = IdxB + 1;
            mA = mLvlData[IdxA];
            mB = mLvlData[IdxB];
            mC = mLvlData[IdxC];
            DevA = DevData[IdxA];
            DevB = DevData[IdxB];
            DevC = DevData[IdxC];

            /* Find bounding box for lower triangle */
            mMin = fmin(fmin(mA,mB),mC);
            mMax = fmax(fmax(mA,mB),mC);
            DevMin = fmin(fmin(DevA,DevB),DevC);
            DevMax = fmax(fmax(DevA,DevB),DevC);

            /* If self is inside bounding box, calc barycentric weights */
            if ((mLvl >= mMin) & (mLvl <= mMax) & (Dev >= DevMin) & (Dev <= DevMax)) {
                SectorWeights = calcBarycentricWeights(mA,DevA,mB,DevB,mC,DevC,mLvl,Dev);

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
            DevB = DevData[IdxB];
            
            /* Find bounding box for upper triangle (Dev bounds don't change) */
            mMin = fmin(fmin(mA,mB),mC);
            mMax = fmax(fmax(mA,mB),mC);

            /* If self is inside bounding box, calc barycentric weights */
            if ((mLvl >= mMin) & (mLvl <= mMax) & (Dev >= DevMin) & (Dev <= DevMax)) {
                SectorWeights = calcBarycentricWeights(mA,DevA,mB,DevB,mC,DevC,mLvl,Dev);

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

            j++; /* Move to next Dev for this mLvl in the data */
         }
         i++; /* Move to next mLvl in the data, resetting Dev */
    }

    /* Store the final xLvl and value in the Out buffers */
    xLvlOut[Gid] = xLvl;
    ValueOut[Gid] = Value;
}


/* This version is for when there is zero medical need shock */
__kernel void doJorgensenDruedahlSimpleFix(
     __global double *mLvlData           /* data on endogenous mLvl */
    ,__global double *pLvlData           /* data on exogenous pLvl */
    ,__global double *ValueData          /* data on value at (m,Dev) */
    ,__global double *cLvlData           /* data on optimal cLvl    */
    ,__global double *mGridDense         /* exogenous grid of mNrm  */
    ,__global double *pGridDense         /* exogenous grid of pLvl */
    ,__global double *pGridDenseAlt      /* alternate grid of pLvl (only for determining mLvlGrid) */
    ,__global double *cLvlOut            /* J-D fixed cLvl to return */
    ,__global double *ValueOut           /* J-D fixed value to return */
    ,__global int *IntegerInputs         /* integers that characterize problem */
    ) {

    /* Initialize some variables for use in the main loop */
    double mA;
    double mB;
    double mC;
    double pA;
    double pB;
    double pC;
    double vA;
    double vB;
    double vC;
    double xA;
    double xB;
    double xC;
    double mMin;
    double mMax;
    double pMin;
    double pMax;
    double vNew = 0.0;
    double cNew;
    double3 SectorWeights = (double3)(0.0,0.0,0.0);
    int IdxA;
    int IdxB;
    int IdxC;

    /* Unpack the integer inputs */
    int mLvlDataDim = IntegerInputs[0];
    int pLvlDataDim = IntegerInputs[1];
    int mGridDenseSize = IntegerInputs[2];
    int pGridDenseSize = IntegerInputs[3];
    int ThreadCount = IntegerInputs[4];

    /* Initialize this thread's id and get this thread's constant (mLvl,pLvl) identity */
    int Gid = get_global_id(0);     /* global thread id */
    if (Gid >= ThreadCount) {
        return;
    }
    int mGridIdx = Gid/pGridDenseSize;
    int pGridIdx = Gid - mGridIdx*pGridDenseSize;
    double pLvl = pGridDense[pGridIdx];
    double pLvlAlt = pGridDense[pGridIdx];
    double mLvl; /* mGridDense has mNrm values */
    if (pLvl > 0.0) {
        mLvl = mGridDense[mGridIdx]*pLvlAlt;
    }
    else {
        mLvl = mGridDense[mGridIdx]*pGridDenseAlt[1];
    }

    /* Initialize xLvl and value for output */
    double cLvl = cLvlOut[Gid];
    double Value = ValueOut[Gid];

    /* Loop over each triangular sector of (mLvl,Dev) from the data */
    int i = 0;
    int j = 0;
    double Low = -0.001;
    double High = 1.001;
    while (i < (mLvlDataDim-1)) {
        j = 0;
        while (j < (pLvlDataDim-1)) {
            /* Get location data for lower triangle */
            IdxA = i*pLvlDataDim + j;
            IdxB = IdxA + pLvlDataDim;
            IdxC = IdxB + 1;
            mA = mLvlData[IdxA];
            mB = mLvlData[IdxB];
            mC = mLvlData[IdxC];
            pA = pLvlData[IdxA];
            pB = pLvlData[IdxB];
            pC = pLvlData[IdxC];

            /* Find bounding box for lower triangle */
            mMin = fmin(fmin(mA,mB),mC);
            mMax = fmax(fmax(mA,mB),mC);
            pMin = fmin(fmin(pA,pB),pC);
            pMax = fmax(fmax(pA,pB),pC);

            /* If self is inside bounding box, calc barycentric weights */
            if ((mLvl >= mMin) & (mLvl <= mMax) & (pLvl >= pMin) & (pLvl <= pMax)) {
                SectorWeights = calcBarycentricWeights(mA,pA,mB,pB,mC,pC,mLvl,pLvl);

                /* If barycentric weights all between 0 and 1, evaluate vNew */
                if ((SectorWeights.x >= Low) & (SectorWeights.y >= Low) & (SectorWeights.z >= Low) & (SectorWeights.x <= High) & (SectorWeights.y <= High) & (SectorWeights.z <= High)) {
                    vA = ValueData[IdxA];
                    vB = ValueData[IdxB];
                    vC = ValueData[IdxC];
                    vNew = SectorWeights.x*vA + SectorWeights.y*vB + SectorWeights.z*vC;

                    /* If vNew is better than current v, replace v and xLvl in Out */
                    if (vNew > Value) {
                        xA = cLvlData[IdxA];
                        xB = cLvlData[IdxB];
                        xC = cLvlData[IdxC];
                        cNew = SectorWeights.x*xA + SectorWeights.y*xB + SectorWeights.z*xC;
                        cLvl = cNew;
                        Value = vNew;
                    }
                }
            } /* End of checking lower triangle */

            /* Get location data for upper triangle (only need to change B) */
            IdxB = IdxA + 1;
            mB = mLvlData[IdxB];
            pB = pLvlData[IdxB];
            
            /* Find bounding box for upper triangle (Dev bounds don't change) */
            mMin = fmin(fmin(mA,mB),mC);
            mMax = fmax(fmax(mA,mB),mC);

            /* If self is inside bounding box, calc barycentric weights */
            if ((mLvl >= mMin) & (mLvl <= mMax) & (pLvl >= pMin) & (pLvl <= pMax)) {
                SectorWeights = calcBarycentricWeights(mA,pA,mB,pB,mC,pC,mLvl,pLvl);

                /* If barycentric weights all between 0 and 1, evaluate vNew */
                if ((SectorWeights.x >= Low) & (SectorWeights.y >= Low) & (SectorWeights.z >= Low) & (SectorWeights.x <= High) & (SectorWeights.y <= High) & (SectorWeights.z <= High)) {
                    vA = ValueData[IdxA];
                    vB = ValueData[IdxB];
                    vC = ValueData[IdxC];
                    vNew = SectorWeights.x*vA + SectorWeights.y*vB + SectorWeights.z*vC;

                    /* If vNew is better than current v, replace v and cLvl in Out */
                    if (vNew > Value) {
                        xA = cLvlData[IdxA];
                        xB = cLvlData[IdxB];
                        xC = cLvlData[IdxC];
                        cNew = SectorWeights.x*xA + SectorWeights.y*xB + SectorWeights.z*xC;
                        cLvl = cNew;
                        Value = vNew;
                    }
                }
            } /* End of checking upper triangle */

            j++; /* Move to next pLvl for this mLvl in the data */
         }
         i++; /* Move to next mLvl in the data, resetting pLvl */
    }

    /* Store the final cLvl and value in the Out buffers */
    cLvlOut[Gid] = cLvl;
    ValueOut[Gid] = Value;
}

