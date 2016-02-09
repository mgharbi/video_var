<<<<<<< HEAD
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

    warper.setVideos(&A,&B);
    uvw = warper.computeWarp();
    // uvw.save("test");
    uvw.copyToMxArray(dimsA[0]*dimsA[1]*dimsA[2]*dimsA[3],outMatrix);
    // uvw.exportSpacetimeMap(params.outputPath, name);
=======
#include "mex.h"
#include "temp.h"

/* The computational routine */
void stwarp(double x, double *y, double *z, mwSize n)
{
    mwSize i;
    /* multiply each element y by x */
    for (i=0; i<n; i++) {
        z[i] = x * y[i];
    }
>>>>>>> 08e370661f7c4bf3290c9d5ccd5d5a000f9f11fc
}

/* The gateway function */
void mexFunction( int nlhs, mxArray *plhs[],
                  int nrhs, const mxArray *prhs[])
{
<<<<<<< HEAD
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
    plhs[0] = mxCreateNumericArray(4, dimsA, mxUINT8_CLASS, mxREAL);

    /* get a pointer to the real data in the output matrix */
    outMatrix = (float*)mxGetData(plhs[0]);

    /* call the computational routine */
    stwarp(dimsA, dimsB, videoA, videoB, outMatrix);
=======
    double multiplier;              /* input scalar */
    double *inMatrix;               /* 1xN input matrix */
    size_t ncols;                   /* size of matrix */
    double *outMatrix;              /* output matrix */

    /* check for proper number of arguments */
    if(nrhs!=2) {
        mexErrMsgIdAndTxt("MyToolbox:stwarp:nrhs","Two inputs required.");
    }
    if(nlhs!=1) {
        mexErrMsgIdAndTxt("MyToolbox:stwarp:nlhs","One output required.");
    }
    /* make sure the first input argument is scalar */
    if( !mxIsDouble(prhs[0]) || 
         mxIsComplex(prhs[0]) ||
         mxGetNumberOfElements(prhs[0])!=1 ) {
        mexErrMsgIdAndTxt("MyToolbox:stwarp:notScalar","Input multiplier must be a scalar.");
    }
    
    /* make sure the second input argument is type double */
    if( !mxIsDouble(prhs[1]) || 
         mxIsComplex(prhs[1])) {
        mexErrMsgIdAndTxt("MyToolbox:stwarp:notDouble","Input matrix must be type double.");
    }
    
    /* check that number of rows in second input argument is 1 */
    if(mxGetM(prhs[1])!=1) {
        mexErrMsgIdAndTxt("MyToolbox:stwarp:notRowVector","Input must be a row vector.");
    }
    
    /* get the value of the scalar input  */
    multiplier = mxGetScalar(prhs[0]);

    /* create a pointer to the real data in the input matrix  */
    inMatrix = mxGetPr(prhs[1]);

    /* get dimensions of the input matrix */
    ncols = mxGetN(prhs[1]);

    /* create the output matrix */
    plhs[0] = mxCreateDoubleMatrix(1,(mwSize)ncols,mxREAL);

    /* get a pointer to the real data in the output matrix */
    outMatrix = mxGetPr(plhs[0]);

    /* call the computational routine */
    stwarp(multiplier,inMatrix,outMatrix,(mwSize)ncols);
>>>>>>> 08e370661f7c4bf3290c9d5ccd5d5a000f9f11fc
}
