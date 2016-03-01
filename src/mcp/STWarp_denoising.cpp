/* --------------------------------------------------------------------------
 * File:    STWarp_denoising.cpp
 * Author:  Michael Gharbi <gharbi@mit.edu>
 * Created: 2014-04-08
 * --------------------------------------------------------------------------
 * 
 * 
 * 
 * ------------------------------------------------------------------------*/


#include "mcp/STWarp.hpp"

template <class T>
void STWarp<T>::denoiseWarpingField( WarpingField<T> &warpField) {

   if( params.medfiltSize <= 0) {
       if(params.verbosity >0) {
            printf("    - no median filtering.\n");
       }
       return;
   } 

    WarpingField<T> orgWarpField(warpField);
    warpField.reset(0);

    if(params.verbosity >0) {
        printf("    - median filtering...");
    }
    // Edge agnostic median filtering
    vector<int> filtSize(2);
    filtSize[0] = params.medfiltSize; filtSize[1] = params.medfiltSize;
    WarpingField<T> filtered(warpField.size());
    VideoProcessing::medfilt2( orgWarpField, filtered, filtSize );
    warpField.copy(filtered);

    int nDataChan = 1;
    if( params.useColor ){
        nDataChan = 3;
    }

    if( !params.useAdvancedMedian ){
        if(params.verbosity >0) {
            printf("done.\n");
        }
        return;
    }

    if(params.verbosity >0) {
        printf("advanced...");
    }

    int height  = warpField.getHeight();
    int width   = warpField.getWidth();
    int nFrames = warpField.frameCount();
    int nVoxels = videoA->voxelCount();

    int window_s_hsz = 7;
    int window_t_hsz = 0;

    T sig_st = 7;
    T sig_color = 7;

    // TODO: check minimal image size

    int wVoxels = (2*window_s_hsz+1);
    wVoxels *= wVoxels; wVoxels *= 2*window_t_hsz+1;

    // Compute spatial weights
    vector<T> w_spacetime( wVoxels );
    for (int k = 0; k < 2*window_t_hsz+1; ++k)
        for (int j = 0; j < 2*window_s_hsz+1; ++j)
            for (int i = 0; i < 2*window_s_hsz+1; ++i)
    {
        int idx = i + (2*window_s_hsz+1)*( j+ (2*window_s_hsz+1)*k );
        T w = (i-window_s_hsz)*(i-window_s_hsz);
        w += (j-window_s_hsz)*(j-window_s_hsz);
        w += (k-window_t_hsz)*(k-window_t_hsz);
        w = exp(-w/(2*sig_st*sig_st));
        w_spacetime[idx] = w;
    }


    // Compute edge mask
    IVideo mask;
    VideoProcessing::edges3D(*videoA,mask,0,nDataChan-1,15);
    int dilateSz[3] = {2, 2, 0};
    VideoProcessing::dilate3D(mask, dilateSz);
    // mask.exportVideo(params.outputPath, "denoiseEdgemap");

    const unsigned char *pMask = mask.dataReader();
    const unsigned char *pA    = videoA->dataReader();

    const T* pOrg = orgWarpField.dataReader();
    T* pNew       = warpField.dataWriter();

    for (int k = 0; k < nFrames; ++k)
        for (int j = 0; j < width; ++j)
            for (int i = 0; i < height; ++i)
    {
        int index = i + height*(j+width*k);
        if( pMask[index] == 0){
            // usual median filter outside of edges
            continue;
        }

        // Color weight
        vector<T> weight( wVoxels );
        vector< vector<T> > values( 3 );
        for (int chan = 0; chan < 3; ++chan) {
            values[chan] = vector<T>(wVoxels);
        }
        T wSum = 0;
        for (int k_w = 0; k_w < 2*window_t_hsz+1; ++k_w)
            for (int j_w = 0; j_w < 2*window_s_hsz+1; ++j_w)
                for (int i_w = 0; i_w < 2*window_s_hsz+1; ++i_w)
        {
            int w_idx = i_w + (2*window_s_hsz+1)*( j_w+ (2*window_s_hsz+1)*k_w );
            
            int iGlob = i+i_w - window_s_hsz;
            int jGlob = j+j_w - window_s_hsz;
            int kGlob = k+k_w - window_t_hsz;
            
            // Reflective boundary conditions
            if( iGlob < 0 ) {
                iGlob *= -1;
            }
            if( jGlob < 0 ) {
                jGlob *= -1;
            }
            if( kGlob < 0 ) {
                kGlob *= -1;
            }
            if( iGlob > height-1 ) {
                iGlob = 2*(height-1) - iGlob;
            }
            if( jGlob > width-1 ) {
                jGlob = 2*(width-1) - jGlob;
            }
            if( kGlob > nFrames-1 ) {
                kGlob = 2*(nFrames-1) - kGlob;
            }

            int global_idx = 
                (iGlob) + 
                height*( (jGlob)+
                width*( kGlob ) );

            T w = 0;
            for (int chan = 0; chan < nDataChan; ++chan) {
                T ww = ( (T) pA[global_idx+chan*nVoxels])-( (T) pA[index+chan*nVoxels]);
                ww *= ww;
                w += ww;
                // Copy warp field value to local var
            }
            for (int chan = 0; chan < 3; ++chan) {
                values[chan][w_idx] = pOrg[global_idx+chan*nVoxels];
            }
            w = exp(-w/(2*sig_color*sig_color));

            weight[w_idx]  = w;
            weight[w_idx] *= w_spacetime[w_idx];

            wSum += weight[w_idx];
        }
        // Normalize
        if(wSum>0){
            for (int w_i = 0; w_i < wVoxels; ++w_i) {
                weight[w_i] /= wSum;
            }
            // Solve weighted median
            // NOTE: chan goes up to 2, we dont process the timemap
            for (int chan = 0; chan < 2; ++chan) {
                T v = weightedMedian( values[chan], weight );
                pNew[index+chan*nVoxels] = v;
            }
        }

    }
    if(params.verbosity >0) {
        fprintf(stderr,"done.\n");
    }
}

template class STWarp<float>;
template class STWarp<double>;
