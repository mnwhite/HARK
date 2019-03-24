# pragma OPENCL EXTENSION cl_khr_fp64 : enable

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


/* Kernel for evaluating a health-copay value function over mLvl, pLvl, Dev */
__kernel void evalValueFunc(
     __global double *mNrmGrid           /* exogenous mNrm grid */
    ,__global double *pLvlGrid           /* exogenous pLvl grid */
    ,__global double *vNvrsZeroShkData   /* data on pseudo-inverse value at (mLvl,pLvl) */
    ,__global double *vNvrsRescaledData  /* data on rescaled value at (mLvl,pLvl,Dev) */
    ,__global double *mLvlQuery          /* market resources query points */
    ,__global double *pLvlQuery          /* permanent income query points */
    ,__global double *DevQuery           /* medical shock deviation query points */
    ,__global double *ValueOut           /* value to return */
    ,__global double *DoubleInputs       /* CRRA, DevMin, DevMax */
    ,__global int *IntegerInputs         /* integers that characterize problem */
    ) {

    /* Unpack the integer inputs */
    int mNrmGridSize = IntegerInputs[0];
    int pLvlGridSize = IntegerInputs[1];
    int DevGridSize = IntegerInputs[2];
    int ThreadCount = IntegerInputs[3];

    /* Unpack the real inputs */
    double CRRA = DoubleInputs[0];
    double DevMin = DoubleInputs[1];
    double DevMax = DoubleInputs[2];
    double DevStep = (DevMax - DevMin)/(convert_double(DevGridSize) - 1.0);

    /* Initialize this thread's id and get its query point */
    int Gid = get_global_id(0);     /* global thread id */
    if (Gid >= ThreadCount) {
        return;
    }
    double mLvl = mLvlQuery[Gid];
    double pLvl = pLvlQuery[Gid];
    double Dev  = DevQuery[Gid];

    /* Initialize some variables */
    int ii;      /* mNrm index for vNvrsRescaled*/
    int ii_alt;  /* mNrm index for vNvrsZeroShk */
    int jj;      /* pLvl index */
    int kk;      /* Dev index */
    int idx;     /* overall index in vNvrsRescaled */
    double mNrm; /* normalized market resources */
    double mLo;  /* market resources at lower gridpoint */
    double mHi;  /* market resources at upper gridpoint */
    double pLo;  /* permanent income at lower gridpoint */
    double pHi;  /* permanent income at upper gridpoint */
    double DevExtra;
    double alpha;/* proportion weight for mNrm on vNvrsRescaled */
    double alpha_alt; /* proportion weight for mNrm on vNvrsZeroShk */
    double beta; /* proportion weight for pLvl */
    double gamma;/* proportion weight for Dev */
    double vLo;  /* lower vNvrs value */
    double vHi;  /* upper vNvrs value */
    double vNvrsZeroShk; /* vNvrs value when MedShk=0 at this (mLvl,pLvl) */
    double vNvrsRescaled;/* vNvrs at (mLvl,pLvl,Dev) rescaled to vNvrsZeroShk */
    double vNvrsLo; /* vNvrs at lower pLvl gridpoint*/
    double vNvrsHi; /* vNvrs at upper pLvl gridpoint */
    double vNvrs;   /* final vNvrs level */
    double Value;   /* final value level */

    /* Find query point's pLvl index and relative weight*/
    jj = findIndex(pLvlGrid,0,pLvlGridSize-1,pLvl);
    jj = max(min(jj,pLvlGridSize-1),1);
    pLo = pLvlGrid[jj-1];
    pHi = pLvlGrid[jj];
    beta = (pLvl - pLo)/(pHi - pLo);

    /* Find query point's Dev index and relative weight */
    DevExtra = Dev - DevMin;
    kk = convert_int(ceil(DevExtra /DevStep));
    kk = max(min(kk,DevGridSize-1),1);
    gamma = (DevExtra - convert_float(kk-1)*DevStep)/DevStep;

    /* Find query point's mNrm index for lower pLvl index */
    mNrm = mLvl/pLo;
    ii = findIndex(mNrmGrid,0,mNrmGridSize-1,mNrm);
    ii = max(min(ii,mNrmGridSize-1),1);

    /* Find query point's mNrm relative weight for lower index */
    mLo = mNrmGrid[ii-1];
    mHi = mNrmGrid[ii];
    alpha = (mNrm-mLo)/(mHi-mLo);
    if (mNrm < mNrmGrid[0]) {
        ii_alt = 1;
        alpha_alt = mNrm/mLo; /* always mLo = mNrmGrid[0] here */
    }
    else {
        ii_alt = ii+1;
        alpha_alt = alpha;
    }

    /* Find query point's vNvrsZeroShk for lower index */
    idx = (jj-1)*(mNrmGridSize+1) + ii_alt;
    vLo = vNvrsZeroShkData[idx - 1];
    vHi = vNvrsZeroShkData[idx];
    vNvrsZeroShk = vLo + alpha_alt*(vHi - vLo);

    /* Find query point's rescaled vNvrs for lower index */
    idx = kk*mNrmGridSize*pLvlGridSize + (jj-1)*mNrmGridSize + ii;
    vLo = vNvrsRescaledData[idx-1];
    vHi = vNvrsRescaledData[idx];
    vNvrsRescaled = gamma*(vLo + alpha*(vHi - vLo));
    idx = idx - mNrmGridSize*pLvlGridSize;
    vLo = vNvrsRescaledData[idx-1];
    vHi = vNvrsRescaledData[idx];
    vNvrsRescaled = vNvrsRescaled + (1.0 - gamma)*(vLo + alpha*(vHi - vLo));

    /* Convert rescaled vNvrs into vNvrs for lower index */
    vNvrsLo = vNvrsZeroShk / (1.0 + exp(vNvrsRescaled));

    /* Find query point's upper index */
    mNrm = mLvl/pHi;
    ii = findIndex(mNrmGrid,0,mNrmGridSize-1,mNrm);
    ii = max(min(ii,mNrmGridSize-1),1);

    /* Find query point's mNrm relative weight for upper index */
    mLo = mNrmGrid[ii-1];
    mHi = mNrmGrid[ii];
    alpha = (mNrm-mLo)/(mHi-mLo);
    if (mNrm < mNrmGrid[0]) {
        ii_alt = 1;
        alpha_alt = mNrm/mLo; /* always mLo = mNrmGrid[0] here */
    }
    else {
        ii_alt = ii+1;
        alpha_alt = alpha;
    }

    /* Find query point's vNvrsZeroShk for upper index */
    idx = (jj)*(mNrmGridSize+1) + ii_alt;
    vLo = vNvrsZeroShkData[idx - 1];
    vHi = vNvrsZeroShkData[idx];
    vNvrsZeroShk = vLo + alpha_alt*(vHi - vLo);

    /* Find query point's rescaled vNvrs for upper index */
    idx = kk*mNrmGridSize*pLvlGridSize + (jj)*mNrmGridSize + ii;
    vLo = vNvrsRescaledData[idx-1];
    vHi = vNvrsRescaledData[idx];
    vNvrsRescaled = gamma*(vLo + alpha*(vHi - vLo));
    idx = idx - mNrmGridSize*pLvlGridSize;
    vLo = vNvrsRescaledData[idx-1];
    vHi = vNvrsRescaledData[idx];
    vNvrsRescaled = vNvrsRescaled + (1.0 - gamma)*(vLo + alpha*(vHi - vLo));

    /* Convert rescaled vNvrs into vNvrs for upper index */
    vNvrsHi = vNvrsZeroShk / (1.0 + exp(vNvrsRescaled));

    /* Calculate weighted vNvrs between pLvls and recurve to value */
    vNvrs = vNvrsLo + beta*(vNvrsHi - vNvrsLo);
    Value = powr(vNvrs,1.0-CRRA)/(1.0-CRRA);
    ValueOut[Gid] = Value;
}
