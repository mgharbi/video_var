#include <cassert>
#include <iostream>

#include "video/Video.hpp"
#include "video/VideoProcessing.hpp"

#include "mcp/NNField.hpp" 

#include "mex.h"

using std::cout;
using std::endl;


/* The computational routine */
void knnfield(const mwSize *dimsA, const mwSize *dimsB, unsigned char* videoA, unsigned char *videoB, NNFieldParams params, int *outMatrix)
{
    IVideo A;
    int dA[4];
    for (int i = 0; i < 4; ++i) {
        dA[i] = dimsA[i];
    }
    A.initFromMxArray(4, dA, videoA);

    IVideo B;
    int dB[4];
    for (int i = 0; i < 4; ++i) {
        dB[i] = dimsB[i];
    }
    B.initFromMxArray(4, dB, videoB);

    NNField field(&A,&B, params);
    Video<int> nnf = field.compute();
    nnf.copyToMxArray(dimsA[0]*dimsA[1]*dimsA[2]*3*params.knn,outMatrix);
}

/* The gateway function */
/**
  * In: video, database, params
  * Out: knn field
  */
void mexFunction( int nlhs, mxArray *plhs[],
                  int nrhs, const mxArray *prhs[])
{
    unsigned char *videoA;       
    unsigned char *videoB;      
    int *outMatrix;  

    // - Checks ------------------------------------------------------------------------------

    /* check for proper number of arguments */
    if(nrhs!=3) {
        mexErrMsgIdAndTxt("MotionComparison:knnfield:nrhs","Three inputs required.");
    }
    if(nlhs!=1) {
        mexErrMsgIdAndTxt("MotionComparison:knnfield:nlhs","One output required.");
    }
    
    /* make sure the first input argument is type uint8 */
    if( !mxIsClass(prhs[0], "uint8") || mxIsComplex(prhs[0])) {
        mexErrMsgIdAndTxt("MotionComparison:knnfield:notFloat","Input matrix must be type uint8.");
    }

    /* make sure the second input argument is type uint8 */
    if( !mxIsClass(prhs[1], "uint8") || mxIsComplex(prhs[1])) {
        mexErrMsgIdAndTxt("MotionComparison:knnfield:notFloat","Input matrix must be type uint8.");
    }

    /* make sure the third input argument is the param struct */
    if( !mxIsStruct(prhs[2]) ) {
        mexErrMsgIdAndTxt("MotionComparison:knnfield:notStruct","Input should be a struct.");
    }

    // - Inputs ------------------------------------------------------------------------------
    
    /* create a pointer to the real data in the input matrix  */
    videoA = (unsigned char*)mxGetData(prhs[0]);
    videoB = (unsigned char*)mxGetData(prhs[1]);

    /* get dimensions of the input matrix */
    if( mxGetNumberOfDimensions(prhs[0]) != 4 || mxGetNumberOfDimensions(prhs[1]) != 4) {
        mexErrMsgIdAndTxt("MotionComparison:knnfield:invalidDim","Input matrices must be 4-dimensional.");
    }

    const mwSize *dimsA = mxGetDimensions(prhs[0]);
    const mwSize *dimsB = mxGetDimensions(prhs[1]);

    for (int i = 0; i < 2; ++i) {
        if(dimsA[i] != dimsB[i]) {
            mexErrMsgIdAndTxt("MotionComparison:knnfield:sizeMismatch","Inputs must have the same spatial dimensions");
        }
    }

    // parse params
    int nfields = mxGetNumberOfFields(prhs[2]);

    if(mxGetNumberOfElements(prhs[2]) != 1) {
        mexErrMsgIdAndTxt("MotionComparison:knnfield:wrongNumberOfStructElem","Param struct must have one element");
    }

    NNFieldParams params;
    for (int i = 0; i < nfields; ++i)
    {
        mxArray *current = mxGetFieldByNumber(prhs[2],0,i);
        const char* field_name = mxGetFieldNameByNumber(prhs[2],i);
        if(current == nullptr) {
            cout << i << " is empty field" << endl;
        }

        if(strcmp(field_name, "propagation_iterations") == 0) {
            double val;
            memcpy(&val, mxGetData(current), mxGetElementSize(current));
            params.propagation_iterations = val;
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

    // - Outputs -----------------------------------------------------------------------------
    

    /* create the output matrix */
    mwSize outSize[4];
    outSize[0] = dimsA[0];
    outSize[1] = dimsA[1];
    outSize[2] = dimsA[2];
    outSize[3] = 3*params.knn;
    plhs[0] = mxCreateNumericArray(4, outSize, mxINT32_CLASS, mxREAL);


    /* get a pointer to the real data in the output matrix */
    outMatrix = (int*)mxGetData(plhs[0]);

    // ---------------------------------------------------------------------------------------

    /* call the computational routine */
    knnfield(dimsA, dimsB, videoA, videoB, params, outMatrix);
}
