/* --------------------------------------------------------------------------
 * File:    warpingIteration.cpp
 * Author:  Michael Gharbi <gharbi@mit.edu>
 * Created: 2014-01-29
 * --------------------------------------------------------------------------
 * 
 * Implementation belonging to the STWarp class
 * 
 * ------------------------------------------------------------------------*/


#include "mcp/STWarp.hpp"

/** Warp videoB towards A and perfom an iteration to update duvw.
 * @see multiscaleIteration
 * @param[in] warpField current estimate of the total warpField
 * @param[in] Bx warped derivative of B
 * @param[in] By warped derivative of B
 * @param[in] Bt warped derivative of B
 * @param[in] C A-B
 * @param[inout] dWarpField incremental update on the warping field;
 */
template <class T>
void STWarp<T>::warpingIteration( const WarpingField<T> &warpField,
                                  const Video<T> &Bx, 
                                  const Video<T> &By, 
                                  const Video<T> &Bt, 
                                  const Video<T> &C, 
                                  WarpingField<T> &dWarpField) {

    int height     = dimensions[0];
    int width      = dimensions[1];
    int nFrames    = dimensions[2];
    // int nChannels  = dimensions[3];

    fprintf(stderr, "    - preparing linear system...");
    // Differentiate (u,v,w)+d(u,v,w) in x,y,t
    WarpingField<T> warpDX(warpField.size());
    WarpingField<T> warpDY(warpField.size());
    WarpingField<T> warpDT(warpField.size());
    WarpingField<T> warpTotal(warpField);
    warpTotal.add(dWarpField);


    VideoProcessing::dx(warpTotal,warpDX);
    VideoProcessing::dy(warpTotal,warpDY);
    VideoProcessing::dt(warpTotal,warpDT);

    // Compute smoothness cost
    Video<T> smoothCost(height, width, nFrames, 9);
    Video<T> lapl(warpField.size());
    computeSmoothCost(warpDX, warpDY, warpDT, warpField, smoothCost, lapl);

    computeOcclusion(warpTotal, C, occlusion);

    // Compute data cost
    Video<T> dataCost;
    
    if(params.useFeatures){
        dataCost = Video<T>(height,width,nFrames,2);
    }else{
        dataCost = Video<T>(height,width,nFrames,1);
    }
    computeDataCost(Bx, By, Bt, C, dWarpField, occlusion, dataCost);

    // Prepare system
    Video<T> CBx(height,width,nFrames,1);
    Video<T> CBy(height,width,nFrames,1);
    Video<T> CBt(height,width,nFrames,1);
    Video<T> Bx2(height,width,nFrames,1);
    Video<T> By2(height,width,nFrames,1);
    Video<T> Bt2(height,width,nFrames,1);
    Video<T> Bxy(height,width,nFrames,1);
    Video<T> Bxt(height,width,nFrames,1);
    Video<T> Byt(height,width,nFrames,1);
    prepareLinearSystem(Bx, By, Bt, C, lapl, dataCost, 
                        CBx, CBy, CBt, Bx2, By2, Bt2,
                        Bxy, Bxt, Byt);
    fprintf(stderr, "done.\n");

    fprintf(stderr, "    - SOR...");
    sor( dataCost, smoothCost, lapl,
         CBx, CBy, CBt, Bx2,
         By2, Bt2, Bxy, Bxt, Byt, dWarpField);
    fprintf(stderr, "done.\n");
}

template class STWarp<float>;
template class STWarp<double>;
