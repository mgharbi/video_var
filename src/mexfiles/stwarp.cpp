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
void stwarp(const mwSize *dimsA, const mwSize *dimsB, unsigned char* videoA, unsigned char *videoB, float *outMatrix)
{
    // TODO:
    // - pass param struct
    // - pass videos
    // - return stwarp
    STWarpParams params;
    WarpingField<precision_t> uvw;
    STWarp<precision_t> warper = STWarp<precision_t>();

    warper.setParams(params);

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

    warper.setVideos(A,B);
    uvw = warper.computeWarp();
    // uvw.save("test");
    uvw.copyToMxArray(dimsA[0]*dimsA[1]*dimsA[2]*dimsA[3],outMatrix);
    // uvw.exportSpacetimeMap(params.outputPath, name);
}

/* The gateway function */
void mexFunction( int nlhs, mxArray *plhs[],
                  int nrhs, const mxArray *prhs[])
{
    unsigned char *videoA;       
    unsigned char *videoB;      
    float *outMatrix;  

    /* check for proper number of arguments */
    if(nrhs!=2) {
        mexErrMsgIdAndTxt("MotionComparison:stwarp:nrhs","Two inputs required.");
    }
    if(nlhs!=1) {
        mexErrMsgIdAndTxt("MotionComparison:stwarp:nlhs","One output required.");
    }
    
    /* make sure the first input argument is type float */
    if( !mxIsClass(prhs[0], "uint8") || mxIsComplex(prhs[0])) {
        mexErrMsgIdAndTxt("MotionComparison:stwarp:notFloat","Input matrix must be type uint8.");
    }

    if( !mxIsClass(prhs[1], "uint8") || mxIsComplex(prhs[1])) {
        mexErrMsgIdAndTxt("MotionComparison:stwarp:notFloat","Input matrix must be type uint8.");
    }
    
    /* create a pointer to the real data in the input matrix  */
    videoA = (unsigned char*)mxGetData(prhs[0]);
    videoB = (unsigned char*)mxGetData(prhs[1]);

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
    cout << "input size: " << dimsA[0] << "x" << dimsA[1] << endl;
    
    /* create the output matrix */
    plhs[0] = mxCreateNumericArray(4, dimsA, mxSINGLE_CLASS, mxREAL);

    /* get a pointer to the real data in the output matrix */
    outMatrix = (float*)mxGetData(plhs[0]);

    /* call the computational routine */
    stwarp(dimsA, dimsB, videoA, videoB, outMatrix);
}
