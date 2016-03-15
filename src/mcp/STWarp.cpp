/* --------------------------------------------------------------------------
 * File:    STWarp.cpp
 * Author:  Michael Gharbi <gharbi@mit.edu>
 * Created: 2014-01-29
 * --------------------------------------------------------------------------
 * 
 * 
 * 
 * ------------------------------------------------------------------------*/


#include "mcp/STWarp.hpp"

template <class T>
STWarp<T>::STWarp() {
    init();
}

template <class T>
void STWarp<T>::init() {
    setDefaultParams();
    dimensions = vector<int>(5,0);
    videoA = nullptr;
    videoB = nullptr;
    initialWarpField = nullptr;
}

template <class T>
void STWarp<T>::setInitialWarpField(const WarpingField<T> &initial) {
    initialWarpField = new WarpingField<T>(initial);
}

template <class T>
STWarp<T>::~STWarp() {
    if( videoA != nullptr ){
        delete videoA;
        videoA = nullptr;
    }
    if( videoB != nullptr ){
        delete videoB;
        videoB = nullptr;
    }
    if( initialWarpField != nullptr ){
        delete initialWarpField;
        initialWarpField = nullptr;
    }
    if(  dimensions.empty() ){
        // TODO: check that
        dimensions.clear();
    }
}


template <class T>
void STWarp<T>::setVideos(const Video<stwarp_video_t> &A, const Video<stwarp_video_t> &B) {
    videoA = new Video<stwarp_video_t>(A);
    videoB = new Video<stwarp_video_t>(B);

    if(videoA->getHeight()!= videoB->getHeight() ||
        videoA->getWidth() != videoB->getWidth() ||
        videoA->frameCount() != videoB->frameCount()){
        if(params.verbosity >0) {
            fprintf(stderr, "Dimensions do not match\n");
        }
    }

    int chan = 3;

    // Use gradient features (requires color)
    if(params.useColor && params.useFeatures) {
        if(params.verbosity > 0) {
            cout << "+ Adding gradient features" << endl;
        }
        videoA->addChannels(3);
        videoB->addChannels(3);

        Video<T> copyA(videoA->size());
        Video<T> copyB(videoA->size());
        copyA.copy(*videoA);
        copyB.copy(*videoB);
        copyA.collapse();
        copyB.collapse();
        Video<T> temp(copyA.size());

        VideoProcessing::dx(copyA, temp);
        videoA->writeChannel(chan, temp);

        VideoProcessing::dy(copyA, temp);
        videoA->writeChannel(chan+1, temp);

        VideoProcessing::dt(copyA, temp);
        videoA->writeChannel(chan+2, temp);

        VideoProcessing::dx(copyB, temp);
        videoB->writeChannel(chan, temp);

        VideoProcessing::dy(copyB, temp);
        videoB->writeChannel(chan+1, temp);

        VideoProcessing::dt(copyB, temp);
        videoB->writeChannel(chan+2, temp);
    }

    // TODO: nullptr
    VideoSize sz  = videoA->size();
    dimensions[0] = sz.height;
    dimensions[1] = sz.width;
    dimensions[2] = sz.nFrames;
    dimensions[3] = sz.nChannels;

    if(params.verbosity > 0) {
        printf("done.\n");
    }
}

template <class T>
void STWarp<T>::buildPyramid(vector<vector<int> > pyrSizes,
        vector<Video<stwarp_video_t>*> &pyramidA, 
        vector<Video<stwarp_video_t>*> &pyramidB
        ) const{
    int n = pyrSizes.size();
    if(params.verbosity > 0) {
        printf("+ Building ST-pyramids with %d levels...",n);
    }
    pyramidA[0] = videoA;
    pyramidB[0] = videoB;
    Video<stwarp_video_t> copy;
    for (int i = 1; i < n; ++i) {
        pyramidA[i] = 
            new Video<stwarp_video_t>(pyrSizes[i][0],
            pyrSizes[i][1],pyrSizes[i][2],dimensions[3]);
        pyramidB[i] = 
            new Video<stwarp_video_t>(pyrSizes[i][0],
            pyrSizes[i][1],pyrSizes[i][2],dimensions[3]);

        // Lowpass and downsample
        copy.copy(*pyramidA[i-1]);
        VideoProcessing::resize(copy,pyramidA[i]);
        copy.copy(*pyramidB[i-1]);
        VideoProcessing::resize(copy,pyramidB[i]);
    }
    if(params.verbosity >0) {
        printf("done.\n");
    }
}


template <class T>
WarpingField<T> STWarp<T>::computeWarp() {
    if(params.verbosity >0) {
        printf("=== Computing warping field for %s, size %dx%dx%d(%d) ===\n",
            params.name.c_str(),
            dimensions[1],
            dimensions[0],
            dimensions[2],
            videoA->channelCount());
    }

    if(params.verbosity >0) {
        if(typeid(T)==typeid(float)) {
            printf("+ Single-precision computation\n");
        } else{
            printf("+ Double-precision computation\n");
        }
        if(params.bypassTimeWarp){
            printf("+ By-passing timewarp map\n");
        }
        printf("+ Regularizing lambda [%5f,%5f,%5f,%5f]\n",
                params.lambda[0],
                params.lambda[1],
                params.lambda[2],
                params.lambda[3]);
    }
    
    // Get dimensions of the pyramid levels
    vector<vector<int> > pyrSizes = getPyramidSizes();
    int nLevels = pyrSizes.size();

    // Build Pyramids
    vector<Video<stwarp_video_t>*> pyramidA(nLevels);
    vector<Video<stwarp_video_t>*> pyramidB(nLevels);
    buildPyramid(pyrSizes,pyramidA,pyramidB);

    WarpingField<T> warpField;
    if(initialWarpField) {
        warpField = *initialWarpField;
    } else {
        if(params.verbosity >0) {
            printf("+ Generating initial warp field\n");
        }
        warpField = WarpingField<T>(dimensions[0], dimensions[1], 
                dimensions[2], 3);
    }

    for (int i = nLevels-1; i >= 0 ; --i) {
        videoA = pyramidA[i];
        videoB = pyramidB[i];

        // update dimensions
        this->dimensions = videoA->dimensions();
        if(params.verbosity >0) {
            printf("+ Multiscale level %02d: %dx%dx%d (B:%d)\n", i+1,dimensions[1],
                dimensions[0],dimensions[2],videoB->frameCount());
        }

        // resample warping field
        resampleWarpingField(warpField,pyrSizes[i]);

        // computeUVW
        multiscaleIteration(warpField);

        if(params.verbosity >0) {
            printf("  x[%.4f, %.4f] ", warpField.min(0), warpField.max(0));
            printf("  y[%.4f, %.4f] ", warpField.min(1), warpField.max(1));
            printf("  t[%.4f, %.4f]\n", warpField.min(2), warpField.max(2));
        }

        // Cleanup allocated videos
        if (i != 0) {
            if( videoA != nullptr ){
                delete videoA;
                videoA = nullptr;
            }
            if( videoB != nullptr ){
                delete videoB;
                videoB = nullptr;
            }
        }
    }

    return warpField;
}

template <class T>
void STWarp<T>::multiscaleIteration(WarpingField<T> &warpField) {
    for (int warpIter = 0; warpIter < params.warpIterations; ++warpIter) {
        if(params.verbosity > 1) {
            printf("  - warp iteration %02d\n",warpIter+1);
        }

        // Get derivatives
        if(params.verbosity > 1) {
            printf("    - computing derivatives...");
        }
        Video<T> Bx(videoB->size());
        Video<T> By(videoB->size());
        Video<T> Bt(videoB->size());
        Video<T> C(videoB->size());
        computePartialDerivatives(warpField,Bx,By,Bt,C);

        if(params.verbosity >1) {
            printf("done.\n");
        }

        // marginal warpField update
        WarpingField<T> dWarpField = WarpingField<T>(warpField.size());

        warpingIteration(warpField, Bx, By, Bt, C, dWarpField);

        if (params.limitUpdate) {
            if(params.verbosity > 1) {
                printf("    - limiting warp update to [-1,1]");
            }
            dWarpField.clamp(-1,1);
        }

        // w <- w + dw
        warpField.add(dWarpField);

        denoiseWarpingField(warpField);

    } // end of warp iteration
}


/**
 * Compute the partial derivatives of the current warping field
 * @param[in] warpField source warping field
 * @param[out] Bx x-derivative
 * @param[out] By y-derivative
 * @param[out] Bt t-derivative
 * @param[out] C A-B
 */
template <class T>
void STWarp<T>::computePartialDerivatives( const WarpingField<T> &warpField,
                                        Video<T> &Bx,
                                        Video<T> &By,
                                        Video<T> &Bt,
                                        Video<T> &C) {
    VideoProcessing::dx(*videoB,Bx,true);
    VideoProcessing::dy(*videoB,By,true);
    VideoProcessing::dt(*videoB,Bt,true);

    Video<stwarp_video_t> warpedB(videoA->size());

    C.copy(*videoA);
    VideoProcessing::backwardWarp(*videoB,warpField,warpedB);
    C.subtract(warpedB);
    // TODO: out of bounds set to 0

    Video<T> temp(videoA->size());
    VideoProcessing::backwardWarp(Bx,warpField,temp);
    Bx.copy(temp);

    temp.reset(0);
    VideoProcessing::backwardWarp(By,warpField,temp);
    By.copy(temp);

    temp.reset(0);
    if( !params.bypassTimeWarp ){
        VideoProcessing::backwardWarp(Bt,warpField,temp);
    }
    Bt.copy(temp);
}

/**
 * Compute the sizes in the Space-Time pyramid.
 */
template <class T>
vector<vector<int> > STWarp<T>::getPyramidSizes() const{
    // TODO: for now we assume inputs have same number of frames
    int spatial_extent = max(dimensions[0], dimensions[1]);
    bool isTimeLonger = spatial_extent < dimensions[2];

    // Reduce the longuest dimension first

    int pyrLevels[3];
    if (params.autoLevels) {
        if(params.verbosity > 0) {
            printf("+ Automatic pyramid levels\n");
        }
        for (int i = 0; i < 2; ++i) {
            pyrLevels[i] = 1 + ( 
                    log( ( (double)dimensions[i] ) /params.minPyrSize )/
                    log( params.pyrSpacing )
            );
        }
        pyrLevels[2] =  1 + ( 
                log( ( (double)dimensions[2] )/params.minPyrSize  )/
                log( params.pyrSpacing )
        );
    }else{
        if(params.verbosity > 0) {
            printf("+ Manual pyramid levels: %d\n", params.pyrLevels);
        }
        for (int i = 0; i < 2; ++i) {
            pyrLevels[i] = params.pyrLevels;
        }
        pyrLevels[2] = 1;
    }

    int nPyrLevelsSpace = max(min(pyrLevels[0], pyrLevels[1]),1);
    int nPyrLevelsTime  = max(pyrLevels[2],1);
    int nPyrLevels      = max(nPyrLevelsTime, nPyrLevelsSpace);
    int diff            = abs(nPyrLevelsTime - nPyrLevelsSpace);

    vector<vector<int> > pyrSizes(nPyrLevels);
    vector<int> currDim = dimensions;
    pyrSizes[0] = currDim;

    for (int i = 1; i < nPyrLevels; ++i) {
        if(isTimeLonger && i < diff ) { // reduce time dim
            currDim[2] = ceil( currDim[2]/params.pyrSpacing );
        } else if(!isTimeLonger && i < diff) { // reduce space dims
            currDim[0] = ceil( currDim[0]/params.pyrSpacing );
            currDim[1] = ceil( currDim[1]/params.pyrSpacing );
        } else { // reduce all
            currDim[0] = ceil( currDim[0]/params.pyrSpacing );
            currDim[1] = ceil( currDim[1]/params.pyrSpacing );
            currDim[2] = ceil( currDim[2]/params.pyrSpacing );
        }
        pyrSizes[i] = currDim;
    }

    return pyrSizes;
}

/**
 * Resample the warping field between pyramid levels
 * @param[inout] warpField warping field to be resampled
 * @params[in] dims target dimensions for the resampling
 */
template <class T>
void STWarp<T>::resampleWarpingField(WarpingField<T>& warpField, vector<int> dims) const {
    // Resize
    WarpingField<T> rWarpField = WarpingField<T>(dims[0],dims[1],dims[2],3);
    VideoProcessing::resize(warpField,&rWarpField);

    // Scale values
    vector<int> oldDims = warpField.dimensions();
    for (int i = 0; i < 3; ++i) {
        T ratio = ( (T) dims[i])/oldDims[i];
        rWarpField.scalarMultiplyChannel(ratio,i);
    }

    warpField = rWarpField;
}

/**
 * Sets default parameters for the correspondence algorithm.
 */
template <class T>
void STWarp<T>::setDefaultParams() {
    params = STWarpParams();
}

/**
 * Parameters accessor
 */
template <class T>
void STWarp<T>::setParams(STWarpParams params) {
    this->params = params;
}

/**
 * Parameters accessor
 */
template <class T>
STWarpParams STWarp<T>::getParams() {
    return params;
}


template class STWarp<float>;
template class STWarp<double>;
