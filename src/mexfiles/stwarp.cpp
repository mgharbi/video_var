#include <cassert>
#include <iostream>

#include "video/Video.hpp"
#include "video/VideoProcessing.hpp"

#include "mcp/STWarp.hpp" 
#include "mcp/Renderer.hpp"
#include "mcp/WarpingField.hpp"

#include "mex.h"

using std::cout;
using std::endl;

typedef float precision_t;

/* The computational routine */
void stwarp(const mwSize *dimsA, const mwSize *dimsB, stwarp_video_t * videoA, stwarp_video_t *videoB, float *outMatrix, STWarpParams params, const float* initWarp)
{
    // TODO:
    // - pass param struct
    // - pass videos
    // - return stwarp
    WarpingField<precision_t> uvw;
    STWarp<precision_t> warper = STWarp<precision_t>();

    warper.setParams(params);

    Video<stwarp_video_t> A;
    int dA[4];
    for (int i = 0; i < 4; ++i) {
        dA[i] = dimsA[i];
    }
    A.initFromMxArray(4, dA, videoA);

    Video<stwarp_video_t> B;
    int dB[4];
    for (int i = 0; i < 4; ++i) {
        dB[i] = dimsB[i];
    }
    B.initFromMxArray(4, dB, videoB);


    warper.setVideos(A,B);

    if(initWarp) {
        WarpingField<float> warp;
        int dWF[4];
        for (int i = 0; i < 3; ++i) {
            dWF[i] = dimsA[i];
        }
        dWF[3] = 3;
        warp.initFromMxArray(4, dWF, initWarp);
        warper.setInitialWarpField(warp);
    }

    uvw = warper.computeWarp();
    uvw.copyToMxArray(dimsA[0]*dimsA[1]*dimsA[2]*dimsA[3],outMatrix);
}

/* The gateway function */
void mexFunction( int nlhs, mxArray *plhs[],
                  int nrhs, const mxArray *prhs[])
{
    // - Checks ------------------------------------------------------------------------------

    /* check for proper number of arguments */
    if(nrhs!=3 && nrhs != 4) {
        mexErrMsgIdAndTxt("MotionComparison:stwarp:nrhs","Three of four inputs required.");
    }
    if(nlhs!=1) {
        mexErrMsgIdAndTxt("MotionComparison:stwarp:nlhs","One output required.");
    }
    
    /* make sure the first input argument is type float */
    if( !mxIsClass(prhs[0], "single") || mxIsComplex(prhs[0])) {
        mexErrMsgIdAndTxt("MotionComparison:stwarp:notFloat","Input matrix must be type float.");
    }

    /* make sure the second input argument is type float */
    if( !mxIsClass(prhs[1], "single") || mxIsComplex(prhs[1])) {
        mexErrMsgIdAndTxt("MotionComparison:stwarp:notFloat","Input matrix must be type float.");
    }

    /* make sure the third input argument is the param struct */
    if( !mxIsStruct(prhs[2]) ) {
        mexErrMsgIdAndTxt("MotionComparison:stwarp:notStruct","Input should be a struct.");
    }

    if( nrhs == 4 && (!mxIsClass(prhs[1], "single") || mxIsComplex(prhs[1])) ) {
        mexErrMsgIdAndTxt("MotionComparison:stwarp:notFloat","Input matrix must be type float.");
    }

    // - Inputs ------------------------------------------------------------------------------
    
    /* create a pointer to the real data in the input matrix  */
    stwarp_video_t *videoA = (stwarp_video_t*)mxGetData(prhs[0]);
    stwarp_video_t *videoB = (stwarp_video_t*)mxGetData(prhs[1]);

    /* get dimensions of the input matrix */
    if( mxGetNumberOfDimensions(prhs[0]) != 4 || mxGetNumberOfDimensions(prhs[1]) != 4) {
        mexErrMsgIdAndTxt("MotionComparison:stwarp:invalidDim","Input matrices must be 4-dimensional.");
    }

    const mwSize *dimsA = mxGetDimensions(prhs[0]);
    const mwSize *dimsB = mxGetDimensions(prhs[1]);

    for (int i = 0; i < 2; ++i) {
        if(dimsA[i] != dimsB[i]) {
            mexErrMsgIdAndTxt("MotionComparison:stwarp:sizeMismatch","Inputs must have the same spatial dimensions");
        }
    }

    // parse params
    int nfields = mxGetNumberOfFields(prhs[2]);

    if(mxGetNumberOfElements(prhs[2]) != 1) {
        mexErrMsgIdAndTxt("MotionComparison:stwarp:wrongNumberOfStructElem","Param struct must have one element");
    }

    float *initWarp = nullptr; 
    if(nrhs == 4) {
        const mwSize *dimsWF = mxGetDimensions(prhs[0]);
        for (int i = 0; i < 3; ++i) {
            if(dimsA[i] != dimsWF[i]) {
                mexErrMsgIdAndTxt("MotionComparison:stwarp:sizeMismatch","Inputs must have the same spatial dimensions");
            }
        }
        if(3 != dimsWF[3]) {
            mexErrMsgIdAndTxt("MotionComparison:stwarp:sizeMismatch","Initial warping field should have 3 channels");
        }
        initWarp = (float*)mxGetData(prhs[3]);
    }

    STWarpParams params;
    for (int i = 0; i < nfields; ++i)
    {
        mxArray *current = mxGetFieldByNumber(prhs[2],0,i);
        const char* field_name = mxGetFieldNameByNumber(prhs[2],i);
        if(current == nullptr) {
            cout << i << " is empty field" << endl;
        }
        if(strcmp(field_name, "verbosity") == 0) {
            double val;
            memcpy(&val, mxGetData(current), mxGetElementSize(current));
            params.verbosity = val;
        }
        if(strcmp(field_name, "reg_spatial_uv") == 0) {
            double val;
            memcpy(&val, mxGetData(current), mxGetElementSize(current));
            params.lambda[0] = val;
        }
        if(strcmp(field_name, "reg_temporal_uv") == 0) {
            double val;
            memcpy(&val, mxGetData(current), mxGetElementSize(current));
            params.lambda[1] = val;
        }
        if(strcmp(field_name, "reg_spatial_w") == 0) {
            double val;
            memcpy(&val, mxGetData(current), mxGetElementSize(current));
            params.lambda[2] = val;
        }
        if(strcmp(field_name, "reg_temporal_w") == 0) {
            double val;
            memcpy(&val, mxGetData(current), mxGetElementSize(current));
            params.lambda[3] = val;
        }
        if(strcmp(field_name, "reg_temporal_w") == 0) {
            double val;
            memcpy(&val, mxGetData(current), mxGetElementSize(current));
            params.lambda[3] = val;
        }
        if(strcmp(field_name, "use_color") == 0) {
            bool val;
            memcpy(&val, mxGetData(current), mxGetElementSize(current));
            params.useColor = val;
        }
        if(strcmp(field_name, "use_gradients") == 0) {
            bool val;
            memcpy(&val, mxGetData(current), mxGetElementSize(current));
            params.useFeatures = val;
        }
        if(strcmp(field_name, "limit_update") == 0) {
            bool val;
            memcpy(&val, mxGetData(current), mxGetElementSize(current));
            params.limitUpdate = val;
        }
        if(strcmp(field_name, "media_filter_size") == 0) {
            double val;
            memcpy(&val, mxGetData(current), mxGetElementSize(current));
            params.medfiltSize = val;
        }
        if(strcmp(field_name, "use_advanced_median") == 0) {
            bool val;
            memcpy(&val, mxGetData(current), mxGetElementSize(current));
            params.useAdvancedMedian = val;
        }
        if(strcmp(field_name, "pyramid_levels") == 0) {
            double val;
            memcpy(&val, mxGetData(current), mxGetElementSize(current));
            if(val>0) {
                params.pyrLevels = val;
                params.autoLevels = false;
            } else {
                params.autoLevels = true;
            }
        }
        if(strcmp(field_name, "min_pyramid_size") == 0) {
            double val;
            memcpy(&val, mxGetData(current), mxGetElementSize(current));
            params.minPyrSize = val;
        }
        if(strcmp(field_name, "solver_iterations") == 0) {
            double val;
            memcpy(&val, mxGetData(current), mxGetElementSize(current));
            params.solverIterations = val;
        }
        if(strcmp(field_name, "warping_iterations") == 0) {
            double val;
            memcpy(&val, mxGetData(current), mxGetElementSize(current));
            params.warpIterations = val;
        }
        if(strcmp(field_name, "pyramid_spacing") == 0) {
            double val;
            memcpy(&val, mxGetData(current), mxGetElementSize(current));
            params.pyrSpacing = val;
        }
        if(strcmp(field_name, "regularization_mode") == 0) {
            double val;
            memcpy(&val, mxGetData(current), mxGetElementSize(current));
            params.decoupleRegularization = val;
        }
    }

    // - Outputs -----------------------------------------------------------------------------
    
    /* create the output matrix */
    float *outMatrix;  
    plhs[0] = mxCreateNumericArray(4, dimsA, mxSINGLE_CLASS, mxREAL);

    /* get a pointer to the real data in the output matrix */
    outMatrix = (float*)mxGetData(plhs[0]);

    // ---------------------------------------------------------------------------------------

    /* call the computational routine */
    stwarp(dimsA, dimsB, videoA, videoB, outMatrix, params, initWarp);
}
