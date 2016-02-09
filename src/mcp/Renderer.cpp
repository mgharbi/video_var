/* --------------------------------------------------------------------------
 * File:    Renderer.cpp
 * Author:  Michael Gharbi <gharbi@mit.edu>
 * Created: 2014-01-29
 * --------------------------------------------------------------------------
 * 
 * 
 * 
 * ------------------------------------------------------------------------*/

#include "mcp/Renderer.hpp"

template <class T>
IVideo Renderer<T>::render(const IVideo& videoA, 
                           const IVideo& videoB, 
                           const WarpingField<T> &warpField,
                           double* exageration,
                           WarpingField<T> *flow) {
    printf("+ Rendering [%f,%f,%f]\n",exageration[0],exageration[1],exageration[2]);
    WarpingField<T> uvw(warpField);
    uvw.scalarMultiplyChannel(exageration[0],0);
    uvw.scalarMultiplyChannel(exageration[1],1);
    uvw.scalarMultiplyChannel(exageration[2],2);

    IVideo C(videoA.size());
    C.copy(videoA);
    IVideo warpedB(videoA.size());
    VideoProcessing::backwardWarp(videoB,warpField,warpedB);
    C.subtract(warpedB);
    C.collapse();

    Video<T> cost(warpField.getHeight(),warpField.getWidth(),warpField.frameCount(),1);
    if(flow == NULL){
        cost.copy(C);
        cost.multiply(C);
    }else{
        T* pCost = cost.dataWriter();
        T sig = 2;
        T* pU = uvw.channelWriter(0);
        T* pV = uvw.channelWriter(1);
        for (int i = 0; i < cost.voxelCount(); ++i) {
            pCost[i] = pU[i]*pU[i] + pV[i]*pV[i];
            pCost[i] = exp(-pCost[i]/(2*sig*sig));
        }
        cost.exportVideo(params.outputPath,"costVideo");
    }

    Video<T> mask(warpField.getHeight(),warpField.getWidth(),warpField.frameCount(),1);
    WarpingField<T> outPrefilt(warpField.size());

    printf("  - Forward warping...");
    int splatSz[3] = {params.splatSize,params.splatSize,params.splatSize};
    VideoProcessing::forwardWarp(uvw,uvw,cost,splatSz,outPrefilt,mask);
    WarpingField<T> out(warpField.size());
    vector<int> filtSize(2);
    filtSize[0] = 5; filtSize[1] = 5;
    printf("done\n");

    if(params.renderSmoothing){
        printf("  - Median filtering...");
        VideoProcessing::medfilt2( outPrefilt, out, filtSize );
        printf("done\n");
    }else{
        out.copy(outPrefilt);
    }

    boost::format f("mask_%02d_%02d_%02d");
    f = f % exageration[0] % exageration[1] % exageration[2];
    mask.exportVideo(params.outputPath, f.str());

    out.scalarMultiply(-1);
    f = boost::format("forwardWarpedFlow_%02d_%02d_%02d");
    f = f % exageration[0] % exageration[1] % exageration[2];
    out.exportSpacetimeMap(params.outputPath, f.str());

    IVideo warpedA(videoA.size());
    VideoProcessing::backwardWarp(videoA,out,warpedA);
    int vC = warpedA.voxelCount();
    unsigned char *pW = warpedA.dataWriter();
    const T* pM = mask.dataReader();
    for (int i = 0; i < vC; ++i) {
        for (int k = 0; k < warpedA.channelCount(); ++k) {
            if(pM[i]==0){
                pW[i+k*vC] *= 0;
            }
        }
    }
    return warpedA;
}

#pragma mark - Template instantiations
template class Renderer<float>;
template class Renderer<double>;
