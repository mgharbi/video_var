#include <cassert>
#include <iostream>

#include "video/Video.hpp"
#include "video/VideoProcessing.hpp"

#include "mcp/NNReconstruction.hpp" 

#include "mex.h"

using std::cout;
using std::endl;


/* The computational routine */
void reconstruct_from_knnf(const mwSize *dims, const unsigned char* db, 
        const int32_t *nnf, const float* w, NNReconstructionParams params, uint8_t *outMatrix)
{

    IVideo video_db;
    int d[4];
    for (int i = 0; i < 3; ++i) {
        d[i] = dims[i];
    }
    d[3] = 3;
    video_db.initFromMxArray(4, d, db);

    Video<int32_t> video_nnf;
    int d_nnf[4];
    for (int i = 0; i < 3; ++i) {
        d_nnf[i] = d[i];
    }
    d_nnf[3] = 3*params.knn;
    video_nnf.initFromMxArray(4, d_nnf, nnf);

    Video<float> video_w;
    int d_w[4];
    for (int i = 0; i < 3; ++i) {
        d_w[i] = d[i];
    }
    d_w[3] = params.knn;
    video_w.initFromMxArray(4, d_w, w);

    // IVideo out;
    NNReconstruction recons(&video_db,&video_nnf,&video_w,params);
    IVideo out = recons.reconstruct();

    out.copyToMxArray(d_nnf[0]*d_nnf[1]*d_nnf[2]*3,outMatrix);
}

/* The gateway function */
/**
  */
void mexFunction( int nlhs, mxArray *plhs[],
                  int nrhs, const mxArray *prhs[])
{

    // - Checks ------------------------------------------------------------------------------

    /* check for proper number of arguments */
    if(nrhs!=4) {
        mexErrMsgIdAndTxt("MotionComparison:reconstruct:nrhs","Four inputs required.");
    }
    if(nlhs !=1 ) {
        mexErrMsgIdAndTxt("MotionComparison:reconstruct:nlhs","One outputs required.");
    }
    
    if( !mxIsClass(prhs[0], "uint8") || mxIsComplex(prhs[0])) {
        mexErrMsgIdAndTxt("MotionComparison:reconstrucreconstruct:notUint8","Input matrix must be type uint8.");
    }

    if( !mxIsClass(prhs[1], "int32") || mxIsComplex(prhs[1])) {
        mexErrMsgIdAndTxt("MotionComparison:reconstruct:notInt32","Input matrix must be type int32.");
    }

    if( !mxIsClass(prhs[2], "single") || mxIsComplex(prhs[2])) {
        mexErrMsgIdAndTxt("MotionComparison:reconstruct:notFloat32","Input matrix must be type Float32.");
    }

    /* make sure the third input argument is the param struct */
    if( !mxIsStruct(prhs[3]) ) {
        mexErrMsgIdAndTxt("MotionComparison:reconstruct:notStruct","Input should be a struct.");
    }

    // - Inputs ------------------------------------------------------------------------------
    
    /* create a pointer to the real data in the input matrix  */
    unsigned char *db = (unsigned char*)mxGetData(prhs[0]);
    int32_t *nnf = (int32_t *)mxGetData(prhs[1]);
    float *w = (float *)mxGetData(prhs[2]);

    /* get dimensions of the input matrix */
    if( mxGetNumberOfDimensions(prhs[0]) != 4 || mxGetNumberOfDimensions(prhs[1]) != 4 ) {
        mexErrMsgIdAndTxt("MotionComparison:reconstruct:invalidDim","Input matrices must be 4-dimensional.");
    }

    const mwSize *dims = mxGetDimensions(prhs[1]);
    const mwSize *dims_w = mxGetDimensions(prhs[2]);
    for (int i = 0; i < 3; ++i) {
        if( dims[i] != dims_w[i] ) {
            mexErrMsgIdAndTxt("MotionComparison:reconstruct:invalidDim","NN-field and weight should have the same space-time dimensions.");
        }
    }

    // parse params
    int nfields = mxGetNumberOfFields(prhs[3]);

    if(mxGetNumberOfElements(prhs[3]) != 1) {
        mexErrMsgIdAndTxt("MotionComparison:reconstruct:wrongNumberOfStructElem","Param struct must have one element");
    }

    NNReconstructionParams params;
    for (int i = 0; i < nfields; ++i)
    {
        mxArray *current = mxGetFieldByNumber(prhs[3],0,i);
        const char* field_name = mxGetFieldNameByNumber(prhs[3],i);
        if(current == nullptr) {
            cout << i << " is empty field" << endl;
        }

        if(strcmp(field_name, "patch_size_space") == 0) {
            double val;
            memcpy(&val, mxGetData(current), mxGetElementSize(current));
            params.patch_size_space = val;
        }
        if(strcmp(field_name, "patch_size_time") == 0) {
            double val;
            memcpy(&val, mxGetData(current), mxGetElementSize(current));
            params.patch_size_time = val;
        }
        if(strcmp(field_name, "knn") == 0) {
            double val;
            memcpy(&val, mxGetData(current), mxGetElementSize(current));
            params.knn = val;
        }
        if(strcmp(field_name, "threads") == 0) {
            double val;
            memcpy(&val, mxGetData(current), mxGetElementSize(current));
            params.threads = val;
        }
    }
    if( dims[3] != params.knn*3 ) {
        mexErrMsgIdAndTxt("MotionComparison:reconstruct:invalidDim","NN-field should have 3*knn channels.");
    }
    if( params.knn > 1 && dims_w[3] != params.knn ) {
        mexErrMsgIdAndTxt("MotionComparison:reconstruct:invalidDim","weightmap should have knn channels.");
    }

    // - Outputs -----------------------------------------------------------------------------
    

    /* create the output matrix */
    mwSize outSize[4];
    outSize[0] = dims[0];
    outSize[1] = dims[1];
    outSize[2] = dims[2];
    outSize[3] = 3;
    plhs[0] = mxCreateNumericArray(4, outSize, mxUINT8_CLASS, mxREAL);
    /* get a pointer to the real data in the output matrix */
    uint8_t *outMatrix = (uint8_t*)mxGetData(plhs[0]);

    // ---------------------------------------------------------------------------------------

    /* call the computational routine */
    reconstruct_from_knnf(dims, db, nnf,w, params, outMatrix);
}
