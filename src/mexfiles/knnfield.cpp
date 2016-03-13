#include <cassert>
#include <iostream>

#include "video/Video.hpp"
#include "video/VideoProcessing.hpp"

#include "mcp/NNField.hpp" 

#include "mex.h"

using std::cout;
using std::endl;


/* The computational routine */
void knnfield(const mwSize *dimsA, const mwSize *dimsB, nnf_data_t* videoA, nnf_data_t* videoB, NNFieldParams params, int32_t *outMatrix, nnf_data_t * distMatrix)
{
    Video<nnf_data_t> A;
    int dA[4];
    for (int i = 0; i < 4; ++i) {
        dA[i] = dimsA[i];
    }
    A.initFromMxArray(4, dA, videoA);

    Video<nnf_data_t> B;
    int dB[4];
    for (int i = 0; i < 4; ++i) {
        dB[i] = dimsB[i];
    }
    B.initFromMxArray(4, dB, videoB);

    NNField field(&A,&B, params);
    NNFieldOutput output = field.compute();
    Video<int32_t> &nnf  = output.nnf;
    Video<nnf_data_t> &cost  = output.error;

    const int32_t* pData = nnf.dataReader();
    const nnf_data_t* pCost = cost.dataReader();

    int channel_stride = nnf.voxelCount();
    int in_nn_stride   = 3*channel_stride;
    int out_nn_stride  = 3*channel_stride;


    for (unsigned int idx = 0; idx < dimsA[0]*dimsA[1]*dimsA[2]; ++idx) // each voxel
    {
        for(int k = 0; k < params.knn; ++ k) { // each NN
            for (unsigned int c = 0; c < 3; ++c) { // each warp channel
                outMatrix[idx + c*channel_stride + k*out_nn_stride] = pData[idx + c*channel_stride + k*in_nn_stride];
            }
            assert(outMatrix[idx + 0*channel_stride + k*out_nn_stride] < B.getWidth()-params.patch_size_space+1);
            assert(outMatrix[idx + 1*channel_stride + k*out_nn_stride] < B.getHeight()-params.patch_size_space+1);
            assert(outMatrix[idx + 2*channel_stride + k*out_nn_stride] < B.frameCount()-params.patch_size_time+1);
        }
    }

    if(distMatrix != nullptr) {
        for (unsigned int idx = 0; idx < dimsA[0]*dimsA[1]*dimsA[2]; ++idx) // each voxel
        {
            for(int k = 0; k < params.knn; ++ k) { // each NN
                distMatrix[idx + k*channel_stride] = pCost[idx + k*channel_stride];
            }
        }
    }
}

/* The gateway function */
/**
  * In: video, database, params
  * Out: knn field
  */
void mexFunction( int nlhs, mxArray *plhs[],
                  int nrhs, const mxArray *prhs[])
{

    // - Checks ------------------------------------------------------------------------------

    /* check for proper number of arguments */
    if(nrhs!=3) {
        mexErrMsgIdAndTxt("MotionComparison:knnfield:nrhs","Three inputs required.");
    }
    if(nlhs<1 || nlhs > 2) {
        mexErrMsgIdAndTxt("MotionComparison:knnfield:nlhs","One or two outputs required.");
    }
    
    /* make sure the first input argument is type float */
    if( !mxIsClass(prhs[0], "single") || mxIsComplex(prhs[0])) {
        mexErrMsgIdAndTxt("MotionComparison:knnfield:notFloat","Input matrix must be type float.");
    }

    /* make sure the second input argument is type float */
    if( !mxIsClass(prhs[1], "single") || mxIsComplex(prhs[1])) {
        mexErrMsgIdAndTxt("MotionComparison:knnfield:notFloat","Input matrix must be type float.");
    }

    /* make sure the third input argument is the param struct */
    if( !mxIsStruct(prhs[2]) ) {
        mexErrMsgIdAndTxt("MotionComparison:knnfield:notStruct","Input should be a struct.");
    }

    // - Inputs ------------------------------------------------------------------------------
    
    /* create a pointer to the real data in the input matrix  */
    nnf_data_t *videoA = (nnf_data_t*)mxGetData(prhs[0]);
    nnf_data_t *videoB = (nnf_data_t*)mxGetData(prhs[1]);

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
        if(strcmp(field_name, "verbosity") == 0) {
            double val;
            memcpy(&val, mxGetData(current), mxGetElementSize(current));
            params.verbosity = val;
        }
    }

    // TODO: check dimensions

    // - Outputs -----------------------------------------------------------------------------
    

    /* create the output matrix */
    mwSize outSize[4];
    outSize[0] = dimsA[0];
    outSize[1] = dimsA[1];
    outSize[2] = dimsA[2];
    outSize[3] = 3*params.knn;
    plhs[0] = mxCreateNumericArray(4, outSize, mxINT32_CLASS, mxREAL);
    int32_t *outMatrix = (int32_t*)mxGetData(plhs[0]);

    nnf_data_t *distMatrix = nullptr;
    if(nlhs == 2) {
        mwSize outSize2[4] = {dimsA[0], dimsA[1], dimsA[2], (mwSize) params.knn};
        plhs[1] = mxCreateNumericArray(4, outSize2, mxSINGLE_CLASS, mxREAL);
        distMatrix = (nnf_data_t*)mxGetData(plhs[1]);
    }



    // ---------------------------------------------------------------------------------------

    /* call the computational routine */
    knnfield(dimsA, dimsB, videoA, videoB, params, outMatrix, distMatrix);
}
