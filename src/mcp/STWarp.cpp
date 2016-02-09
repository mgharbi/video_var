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
STWarp<T>::STWarp(fs::path outPath) {
    init();
    params.outputPath = outPath;
}

template <class T>
void STWarp<T>::init() {
    setDefaultParams();
    dimensions = vector<int>(5);
    videoA = NULL;
    videoB = NULL;
    maskA  = NULL;
    flowA  = NULL;
    initialWarpField = NULL;
}

template <class T>
void STWarp<T>::setInitialWarpField(WarpingField<T> initial) {
    initialWarpField = new WarpingField<T>(initial);
}

template <class T>
STWarp<T>::~STWarp() {
    if( flowA != NULL ){
        delete flowA;
        flowA = NULL;
    }
    if( maskA != NULL ){
        delete maskA;
        maskA = NULL;
    }
    if( initialWarpField != NULL ){
        delete initialWarpField;
        initialWarpField = NULL;
    }
    if(  dimensions.empty() ){
        // TODO: check that
        dimensions.clear();
    }
}

// template <class T>
// void STWarp<T>::loadMasks(fs::path pathA, fs::path pathB) {
//     maskA = new IVideo(pathA);
//     maskA->collapse();
// }

template <class T>
void STWarp<T>::setVideos(IVideo *A, IVideo *B) {
    videoA = A;
    videoB = B;

    // if(videoA->getHeight()!= videoB->getHeight() ||
    //     videoA->getWidth() != videoB->getWidth() ||
    //     videoA->frameCount() != videoB->frameCount()){
    //     fprintf(stderr, "Dimensions do not match\n");
    // }

    params.useColor = true;
    params.useFeatures = false;

    int chan = 3;
    // Process only a monochromatic image
    if(!params.useColor) {
        videoA->collapse();
        videoB->collapse();
        chan = 1;
    }

    // Use gradient features (requires color)
    if(params.useColor && params.useFeatures) {
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
        temp.scalarAdd(255);
        temp.scalarMultiply(.5);
        videoA->writeChannel(chan, temp);

        VideoProcessing::dy(copyA, temp);
        temp.scalarAdd(255);
        temp.scalarMultiply(.5);
        videoA->writeChannel(chan+1, temp);

        VideoProcessing::dt(copyA, temp);
        temp.scalarAdd(255);
        temp.scalarMultiply(.5);
        videoA->writeChannel(chan+2, temp);

        VideoProcessing::dx(copyB, temp);
        temp.scalarAdd(255);
        temp.scalarMultiply(.5);
        videoB->writeChannel(chan, temp);

        VideoProcessing::dy(copyB, temp);
        temp.scalarAdd(255);
        temp.scalarMultiply(.5);
        videoB->writeChannel(chan+1, temp);

        VideoProcessing::dt(copyB, temp);
        temp.scalarAdd(255);
        temp.scalarMultiply(.5);
        videoB->writeChannel(chan+2, temp);
    }


    // TODO: nullptr
    VideoSize sz  = videoA->size();
    dimensions[0] = sz.height;
    dimensions[1] = sz.width;
    dimensions[2] = sz.nFrames;
    dimensions[3] = videoB->frameCount();
    dimensions[4] = sz.nChannels;

    occlusion = Video<T>(sz.height, sz.width, sz.nFrames, 1);
    occlusion.reset(1);
    printf("done.\n");
}

template <class T>
void STWarp<T>::thresholdFlow(double thresh) {
    flowA->collapse();
    T* fA = flowA->dataWriter();
    for (int i = 0; i < flowA->elementCount(); ++i) {
        if(fA[i]<thresh){
            fA[i] = 0;
        }else{
            fA[i] = 255;
        }
    }

}

// template <class T>
// void STWarp<T>::computeOpticalFlow() {
//     printf("+ Computing optical flow\n");
//     VideoSize s = videoA->size();
//     flowA = new WarpingField<T>(s.height, s.width, s.nFrames,2);
//
//     int nChannels = s.nChannels;
//     if(params.useFeatures) {
//         nChannels -= 3;
//     }
//
//     DImage im1(s.width,s.height,nChannels);
//     DImage im2(s.width,s.height,nChannels);
//
//     T *pOF = NULL; 
//     const unsigned char *pVideo = NULL;
//     int start_frame = 0;
//     int end_frame = videoA->frameCount()-1;
//     int frame_step = 1;
//     for(int pass = 0; pass < 1; ++pass) {
//         pOF    = flowA->dataWriter();
//         pVideo = videoA->dataReader();
//         // if(pass == 1) {
//         //     start_frame = videoA->frameCount()-1;
//         //     end_frame = 0;
//         //     frame_step = -1;
//         //     pOF    = flowB->dataWriter();
//         // }
//         
//         // Extract pair of frames
//         double *p1 = im1.data();
//         double *p2 = im2.data();
//         for (int frame = start_frame; frame != end_frame ; frame += frame_step) {
//             printf("\r\t%d - %03d/%03d",pass+1,frame+1,videoA->frameCount()-1);
//             fflush(stdout);
//             int indexV = 0, indexI = 0;
//             for (int k = 0; k < nChannels; ++k)
//                 for (int j = 0; j < s.width; ++j)
//                     for (int i = 0; i < s.height; ++i)
//                     {
//                         indexV = i+s.height*(j+ s.width*(frame+s.nFrames*k));
//                         indexI = (i*s.width+j)*nChannels + k;
//                         p1[indexI] = pVideo[indexV];
//
//                         indexV += frame_step*s.height*s.width;
//                         p2[indexI] = pVideo[indexV];
//                     }
//
//             double alpha           = params.lambda[0];
//             double ratio           = 1/params.pyrSpacing;
//             int minWidth           = params.minPyrSize;
//             int nOuterFPIterations = params.warpIterations;
//             int nInnerFPIterations = 1;
//             int nSORIterations     = params.solverIterations;
//             DImage vx,vy,warpI2;
//             // OpticalFlow::Coarse2FineFlow(vx,vy,warpI2,im1,im2,
//             //         alpha,ratio,minWidth,
//             //         nOuterFPIterations,nInnerFPIterations,nSORIterations);
//
//             for (int j = 0; j < s.width; ++j)
//                 for (int i = 0; i < s.height; ++i)
//                 {
//                     indexV      = i+s.height*(j+ s.width*frame);
//                     indexI      = i*s.width+j;
//                     pOF[indexV] = (T) vx[indexI];
//
//                     indexV += s.height*s.width*+s.nFrames;
//                     pOF[indexV] = (T) vy[indexI];
//                 }
//         }
//     }
//     printf("\n");
// }

template <class T>
void STWarp<T>::buildPyramid(vector<vector<int> > pyrSizes,
        vector<IVideo*> &pyramidA, 
        vector<IVideo*> &pyramidB,
        vector<IVideo*> &pyrMaskA,
        vector<WarpingField<T>*> &pyrFlowA
        ) const{
    int n = pyrSizes.size();
    printf("+ Building ST-pyramids with %d levels...",n);
    pyramidA[0] = videoA;
    pyramidB[0] = videoB;
    pyrMaskA[0] = maskA;
    pyrFlowA[0] = flowA;
    IVideo copy;
    WarpingField<T> flowCopy;
    for (int i = 1; i < n; ++i) {
        pyramidA[i] = new IVideo(pyrSizes[i][0],
                pyrSizes[i][1],pyrSizes[i][2],dimensions[4]);
        pyramidB[i] = new IVideo(pyrSizes[i][0],
                pyrSizes[i][1],pyrSizes[i][3],dimensions[4]);
        pyrMaskA[i] = new IVideo(pyrSizes[i][0],
                pyrSizes[i][1],pyrSizes[i][2],dimensions[4]);
        pyrFlowA[i] = new WarpingField<T>(pyrSizes[i][0],
                pyrSizes[i][1],pyrSizes[i][2],2);

        // Lowpass then downsample
        copy.copy(*pyramidA[i-1]);
        VideoProcessing::resize(copy,pyramidA[i]);
        copy.copy(*pyramidB[i-1]);
        VideoProcessing::resize(copy,pyramidB[i]);
        copy.copy(*pyrMaskA[i-1]);
        VideoProcessing::resize(copy,pyrMaskA[i]);

        flowCopy.copy(*pyrFlowA[i-1]);
        VideoProcessing::resize(flowCopy,pyrFlowA[i]);

        // Rescale optical flow
        vector<int> oldDim = pyrFlowA[i-1]->dimensions();
        vector<int> newDim = pyrFlowA[i]->dimensions();
        for (int d = 0; d < 2; ++d) {
            T ratio = ( (T) newDim[d] )/oldDim[d];
            pyrFlowA[i]->scalarMultiplyChannel(ratio,d);
        }
    }
    printf("done.\n");
}

template <class T>
WarpingField<T> STWarp<T>::computeWarp() {
    printf("=== Computing warping field for %s, size %dx%dx%d(%d) ===\n",
            params.name.c_str(),
            dimensions[1],
            dimensions[0],
            dimensions[2],
            videoA->channelCount());


    if(typeid(T)==typeid(float)) {
        printf("+ Single-precision computation\n");
    } else{
        printf("+ Double-precision computation\n");
    }
    if(params.bypassTimeWarp){
        printf("+ By-passing timewarp map\n");
    }
    printf("+ Lambda [%5f,%5f,%5f,%5f]\n",
            params.lambda[0],
            params.lambda[1],
            params.lambda[2],
            params.lambda[3]);
    
    // Build Pyramids
    vector<vector<int> > pyrSizes = getPyramidSizes();
    int n = pyrSizes.size();
    vector<IVideo*> pyramidA(n);
    vector<IVideo*> pyramidB(n);
    vector<IVideo*> pyrMaskA(n);
    vector<WarpingField<T>*> pyrFlowA(n);
    buildPyramid(pyrSizes,pyramidA,pyramidB,pyrMaskA,pyrFlowA);

    WarpingField<T> warpField;
    if( initialWarpField != NULL ){
        printf("+ Using init field\n");
        warpField = *initialWarpField;
        // warpField.scalarMultiplyChannel(0, 0);
        // warpField.scalarMultiplyChannel(0, 1);
    }else {
        printf("+ Generating init field\n");
        warpField = WarpingField<T>(dimensions[0], dimensions[1], 
                dimensions[2], 3);
        // Initialize the time map to be a ramp
        T* pW = warpField.dataWriter();
        int nFramesA = videoA->frameCount();
        int nFramesB = videoB->frameCount();
        int nVoxels = videoA->voxelCount();
        double ratio = ((double) nFramesB - 1) / ((double) nFramesA - 1);
        for(int k = 0; k<dimensions[2];++k)
            for(int j = 0; j<dimensions[1];++j)
                for(int i = 0; i<dimensions[0];++i)
        {
            int index = i + dimensions[0]*( j + dimensions[1]*k );
            pW[index + 2*nVoxels] = (T) ratio*k - k;
        }
    }

    for (int i = n-1; i >= 0 ; --i) {
        videoA = pyramidA[i];
        videoB = pyramidB[i];
        flowA  = pyrFlowA[i];

        // update dimensions
        this->dimensions = videoA->dimensions();
        printf("+ Multiscale level %02d: %dx%dx%d (B:%d)\n", i+1,dimensions[1],
            dimensions[0],dimensions[2],videoB->frameCount());

        // resample warping field
        resampleWarpingField(warpField,pyrSizes[i]);

        // computeUVW
        multiscaleIteration(warpField);

        if( params.debug ){
            boost::format outPyr("%s/pyramid");
            outPyr = outPyr % params.outputPath.c_str();
            boost::format fuvw("%s_uvw%02d");
            warpField.exportSpacetimeMap(outPyr.str(), (fuvw % "video"% i).str());
        }

        // Cleanup allocated videos
        if (i != 0) {
            if(flowA != NULL) {
                delete flowA;
                flowA = NULL;
            }
        }
    }

    return warpField;
}

template <class T>
void STWarp<T>::multiscaleIteration(WarpingField<T> &warpField) {
    for (int warpIter = 0; warpIter < params.warpIterations; ++warpIter) {
        printf("  - warp iteration %02d\n",warpIter+1);

        // Get derivatives
        fprintf(stderr, "    - computing derivatives...");
        Video<T> Bx(videoB->size());
        Video<T> By(videoB->size());
        Video<T> Bt(videoB->size());
        Video<T> C(videoB->size());
        computePartialDerivatives(warpField,Bx,By,Bt,C);
        fprintf(stderr, "done.\n");

        WarpingField<T> dWarpField = WarpingField<T>(warpField.size());

        Video<T> newOcc(dimensions[0], dimensions[1], dimensions[2], 1);
        VideoProcessing::resize(occlusion,&newOcc);
        occlusion = newOcc;
        warpingIteration(warpField, Bx, By, Bt, C, dWarpField);

        if (params.limitUpdate) {
            fprintf(stderr, "    - limit update");
            dWarpField.clamp(-1,1);
        }

        warpField.add(dWarpField);

        denoiseWarpingField(warpField, occlusion);

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
    VideoProcessing::dt(*videoB,Bt,false);
    C.copy(*videoA);
    IVideo warpedB(videoA->size());
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
    int pyrLevels[3];
    if (params.autoLevels) {
        for (int i = 0; i < 2; ++i) {
            pyrLevels[i] = 1 + ( 
                    log( ( (double)dimensions[i] ) /params.minPyrSize )/
                    log( params.pyrSpacing )
            );
        }
        pyrLevels[2] = 1 + ( 
                log( ( (double)min(dimensions[2],dimensions[3]) )/( 10 ) )/
                log( params.pyrSpacing )
        );
        // pyrLevels[2] = 0;
    }else{
        for (int i = 0; i < 3; ++i) {
            pyrLevels[i] = params.pyrLevels;
        }
    }

    int nPyrLevelsSpace = max(min(pyrLevels[0], pyrLevels[1]),1);
    int nPyrLevelsTime = max(pyrLevels[2],0);
    vector<vector<int> > pyrSizes(nPyrLevelsSpace+nPyrLevelsTime);
    vector<int> currDim = dimensions;
    pyrSizes[0] = currDim;
    // Spatial downsizing
    for (int i = 1; i < nPyrLevelsSpace; ++i) {
        for (size_t j = 0; j < 2; ++j) {
            currDim[j] = floor( currDim[j]/params.pyrSpacing );
        }
        pyrSizes[i] = currDim;
    }
    // Additional temporal downsize
    for (int i = nPyrLevelsSpace; i < nPyrLevelsSpace+nPyrLevelsTime; ++i) {
        for (size_t j = 2; j < 4; ++j) {
            currDim[j] = floor( currDim[j]/params.pyrSpacing );
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

/**
 * Load parameters from file
 */
template <class T>
void STWarp<T>::loadParams(fs::path path) {
    params.loadParams(path);
}


#pragma mark - Template instantiations
template class STWarp<float>;
template class STWarp<double>;
