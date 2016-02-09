/* --------------------------------------------------------------------------
 * File:    VideoProcessing.hpp
 * Author:  Michael Gharbi <gharbi@mit.edu>
 * Created: 2014-01-29
 * --------------------------------------------------------------------------
 * 
 * Common functions for video processing.
 * 
 * ------------------------------------------------------------------------*/


#ifndef VIDEOPROCESSING_HPP_JSKF1JAR
#define VIDEOPROCESSING_HPP_JSKF1JAR

<<<<<<< HEAD:include/video/VideoProcessing.hpp
=======
#include "video/Video.hpp"
>>>>>>> 08e370661f7c4bf3290c9d5ccd5d5a000f9f11fc:include/video/VideoProcessing.hpp
#include <cmath>

#include "video/Video.hpp"

class VideoProcessing
{
public:

    template <class T, class T2>
    static void hfiltering(const Video<T> &input, Video<T2> &output,const double* pFilter,int fsize);
    template <class T, class T2>
    static void vfiltering(const Video<T> &input, Video<T2> &output,const double* pFilter,int fsize);
    template <class T, class T2>
    static void tfiltering(const Video<T> &input, Video<T2> &output,const double* pFilter,int fsize);
    template <class T, class T2>
    static void medfilt2(const Video<T> &input, Video<T2> &output,vector<int> sz);

    template <class T, class T2>
    static void edges3D(const Video<T> &input, Video<T2> &edgemap, int channelStart = 0, int channelEnd = -1, float thresh = 20);

    // template <class T, class T2, class T3>
    // static void conv2(const Video<T> &input, Video<T2> &kernel, Video<T3> &output);
    //
    // template <class T, class T2, class T3>
    // static void conv3(const Video<T> &input, Video<T2> &kernel, Video<T3> &output);

    template <class T, class T2>
    static void dx(const Video<T> &input, Video<T2> &dest, bool advancedFilter = false);
    template <class T, class T2>
    static void dy(const Video<T> &input, Video<T2> &dest, bool advancedFilter = false);
    template <class T, class T2>
    static void dt(const Video<T> &input, Video<T2> &dest, bool advancedFilter = false);

    template <class T>
    static void gaussianSmoothing(Video<T>& video,double* sigma,int* fsize);

    template <class T>
    static void dilate3D(Video<T> &video, int* fsize);

    static double inline enforceRange(double val, int maxVal) { return fmin( fmax( val, 0),maxVal-1 ); };

    template <class T1, class T2>
    static inline void trilinearInterpolate(const Video<T1> &input, double *queryPoint, T2 *output);
    template <class T1>
    static void resize(const Video<T1> &input, Video<T1> *const out);

    template <class T1, class T2>
    static void backwardWarp(const Video<T1> &source, const Video<T2> warpField, Video<T1> &target);
    template <class T1, class T2>
    static void backwardWarp( Video<T1> &source, const Video<T2> warpField);
    template <class T1, class T2, class T3>
    static void forwardWarp( const Video<T1> &input,
                             const Video<T2> &warpField,
                             const Video<T2> &cost,
                             int splatSz[3],
                             Video<T1> &output,
                             Video<T3> &mask);
    template <class T1, class T2>
    static void backwardWarp2D(const Video<T1> &source, const Video<T2> warpField, Video<T1> &target, int frameOffset);

    // Visualization tools
    template <class T1>
    static void videoFuse( const Video<T1> &inputA, const Video<T1> &inputB, Video<T1> &output);
    template <class T1>
    static void edgeOverlay( const Video<T1> &inputA, const Video<T1> &inputB, Video<T1> &output);
};


#pragma mark - inlined functions

/**
 * Linearly interpolate in a video volume.
 * @param[in] input source video 
 * @param[in] queryPoint 3-vector of the interpolation coordinate
 * @param[out] output interpolated values for as many channels as in input
 */
template <class T1, class T2>
void VideoProcessing::trilinearInterpolate(const Video<T1> &input, double *queryPoint, T2 *output) {
    vector<int> dims = input.dimensions();
    int nVoxels      = input.voxelCount();
    const T1* pInput = input.dataReader();
    int refPoint[3];
    double delta[3];
    for (size_t i = 0; i < 3; ++i) {
        refPoint[i] = static_cast<int>(queryPoint[i]);
        delta[i]    = fmax( fmin( queryPoint[i] - refPoint[i], 1), 0 );
    }
    // reset output
    for(int l = 0; l < dims[3]; ++l) {
        output[l] = 0;
    }

    int u,v,w, index;
    double weight;
    for (int k = 0; k < 2; ++k)
        for (int j = 0; j < 2; ++j)
            for (int i = 0; i < 2; ++i)
    {
        u = enforceRange(refPoint[0]+i, dims[0]);
        v = enforceRange(refPoint[1]+j, dims[1]);
        w = enforceRange(refPoint[2]+k, dims[2]);

        index = u + dims[0]*( v + dims[1]*w );
        weight = fabs(1-i-delta[0]) * fabs(1-j-delta[1]) * fabs(1-k-delta[2]);
        for(int l = 0; l < dims[3]; ++l) {
            output[l] += weight*pInput[index + l*nVoxels];
        }
    }
}

/**
 * Resize the input video to the target's size
 * @param[in] input source video 
 * @param[out] out resized video
 */
template <class T1>
void VideoProcessing::resize(const Video<T1> &input, Video<T1> * const out) {
    // TODO: check dimensions
    vector<int> dims = input.dimensions();
    vector<int> dimsOut = out->dimensions();
    vector<double> ratio(3);
    ratio[0] = static_cast<double>(dims[0])/dimsOut[0];
    ratio[1] = static_cast<double>(dims[1])/dimsOut[1];
    ratio[2] = static_cast<double>(dims[2])/dimsOut[2];
    T1* pOut = out->dataWriter();
    int nVoxels = out->voxelCount();

    Video<T1> copy(input);
    double sigma[3];
    int fsize[3];
    for (int j = 0; j < 3; ++j) {
        sigma[j] = sqrt(ratio[j])/sqrt(2);
        if( ratio[j] > 1 ) {
            fsize[j] = 2*sigma[j];
        }else{
            fsize[j] = 0;
        }
    }
    VideoProcessing::gaussianSmoothing(copy, sigma, fsize);

    int index;
    double queryPoint[3];
    double* buffer = new double[dims[3]];
    if(buffer == NULL) {
        // TODO: alloc exception
    }
    for (int k = 0; k < dimsOut[2]; ++k){
        queryPoint[2] = k*ratio[2];
        for (int j = 0; j < dimsOut[1]; ++j){
            queryPoint[1] = j*ratio[1];
            for (int i = 0; i < dimsOut[0]; ++i){
                index = i + dimsOut[0]*( j + dimsOut[1]*k );
                queryPoint[0] = i*ratio[0];
                trilinearInterpolate(input,queryPoint,buffer);
                for (int l = 0; l < dims[3]; ++l) {
                    pOut[index + nVoxels*l] = buffer[l];
                }
            }
        }
    }
    delete[] buffer;
}

/**
 * Anisotropic 3D gaussian filtering
 */
template <class T>
void VideoProcessing::gaussianSmoothing(Video<T>& video,double* sigma,int* fsize) {
	double* gFilter[3];

    for (int d = 0; d < 3; ++d) {
        gFilter[d] = new double[fsize[d]*2+1];
        double sum = 0;
        sigma[d] = sigma[d]*sigma[d]*2;
        for(int i  = -fsize[d];i<= fsize[d];i++) {
            gFilter[d][i+fsize[d]]=exp(-(double)(i*i)/sigma[d]);
            sum+=gFilter[d][i+fsize[d]];
        }
        for(int i=0;i<2*fsize[d]+1;i++) {
            gFilter[d][i]/=sum;
        }
    }

	// Filter
    Video<T> buffer(video.size());
    if(fsize[0]>0) {
        hfiltering(video, buffer, gFilter[0], fsize[0]);
    }else {
        buffer.copy(video);
    }
    Video<T> buffer2(video.size());
    if(fsize[1]>0) {
        vfiltering(buffer, buffer2, gFilter[1], fsize[1]);
    }else {
        buffer2.copy(buffer);
    }
    if(fsize[2]>0) {
        tfiltering(buffer2, video, gFilter[2], fsize[2]);
    }else {
        video.copy(buffer2);
    }

    for (int d = 0; d < 3; ++d) {
        delete gFilter[d];
    }
}

/**
 * Perform 1D linear filtering (correlation) on the x-axis
 * @param[in] input source video 
 * @param[in] pFilter odd-sized(2k+1) linear filter
 * @param[in] fsize half-size of the filter (k)
 * @param[out] output filtered video
 */
template <class T, class T2>
void VideoProcessing::hfiltering(const Video<T> &input, Video<T2> &output,const double* pFilter,int fsize) {
    output.reset();
    int index, index2, j2;
    vector<int> dims = input.dimensions();
    int nVoxels      = input.voxelCount();
    double w;
    T2* pOut = output.dataWriter();
    const T* pIn = input.dataReader();

    double* accumulator = new double[dims[3]];
    for (int k = 0; k < dims[2]; ++k)
        for (int j = 0; j < dims[1]; ++j)
            for (int i = 0; i < dims[0]; ++i)
    {
        for (int chan = 0; chan < dims[3]; ++chan) {
            accumulator[chan] = 0;
        }
        index = i + dims[0]*( j + dims[1]*k );
        for (int l = -fsize; l <= fsize; ++l) {
            j2 = enforceRange(j+l,dims[1]);
            index2 = i + dims[0]*( j2 + dims[1]*k );
            w = pFilter[l+fsize];
            for (int chan = 0; chan < dims[3]; ++chan) {
               accumulator[chan] += static_cast<double>(pIn[index2 + chan*nVoxels ])*w;
            }
        }
        for (int chan = 0; chan < dims[3]; ++chan) {
            pOut[index + chan*nVoxels] = static_cast<T2>(accumulator[chan]);
        }
    }
    delete accumulator;
}

/**
 * Perform 1D linear filtering (correlation) on the x-axis
 * @param[in] input source video 
 * @param[in] pFilter odd-sized(2k+1) linear filter
 * @param[in] fsize half-size of the filter (k)
 * @param[out] output filtered video
 */
template <class T, class T2>
void VideoProcessing::vfiltering(const Video<T> &input, Video<T2> &output,const double* pFilter,int fsize) {
    output.reset();
    int index, index2, i2;
    vector<int> dims = input.dimensions();
    int nVoxels      = input.voxelCount();
    double w;
    T2* pOut = output.dataWriter();
    const T* pIn = input.dataReader();

    double* accumulator = new double[dims[3]];
    for (int k = 0; k < dims[2]; ++k)
        for (int j = 0; j < dims[1]; ++j)
            for (int i = 0; i < dims[0]; ++i)
    {
        for (int chan = 0; chan < dims[3]; ++chan) {
            accumulator[chan] = 0;
        }
        index = i + dims[0]*( j + dims[1]*k );
        for (int l = -fsize; l <= fsize; ++l) {
            i2 = enforceRange(i+l,dims[0]);
            index2 = i2 + dims[0]*( j + dims[1]*k );
            w = pFilter[l+fsize];
            for (int chan = 0; chan < dims[3]; ++chan) {
               accumulator[chan] += pIn[index2 + chan*nVoxels ]*w;
            }
        }
        for (int chan = 0; chan < dims[3]; ++chan) {
            pOut[index + chan*nVoxels] = accumulator[chan];
        }
    }
    delete accumulator;
}


/**
 * Perform 1D linear filtering (correlation) on the t-axis
 * @param[in] input source video 
 * @param[in] pFilter odd-sized(2k+1) linear filter
 * @param[in] fsize half-size of the filter (k)
 * @param[out] output filtered video
 */
template <class T,class T2>
void VideoProcessing::tfiltering(const Video<T> &input, Video<T2> &output,const double* pFilter,int fsize) {
    output.reset();
    int index, index2, k2;
    vector<int> dims = input.dimensions();
    int nVoxels      = input.voxelCount();
    double w;
    T2* pOut = output.dataWriter();
    const T* pIn = input.dataReader();

    double* accumulator = new double[dims[3]];
    for (int k = 0; k < dims[2]; ++k)
        for (int j = 0; j < dims[1]; ++j)
            for (int i = 0; i < dims[0]; ++i)
    {
        for (int chan = 0; chan < dims[3]; ++chan) {
            accumulator[chan] = 0;
        }
        index = i + dims[0]*( j + dims[1]*k );
        for (int l = -fsize; l <= fsize; ++l) {
            k2 = enforceRange(k+l,dims[2]);
            index2 = i + dims[0]*( j + dims[1]*k2 );
            w = pFilter[l+fsize];
            for (int chan = 0; chan < dims[3]; ++chan) {
               accumulator[chan] += pIn[index2 + chan*nVoxels ]*w;
            }
        }
        for (int chan = 0; chan < dims[3]; ++chan) {
            pOut[index + chan*nVoxels] = accumulator[chan];
        }
    }
    delete accumulator;
}

/**
 * Perform 2D median filtering on every frame ov video, with reflective boundary
 * conditions
 * @param[in] input source video 
 * @param[out] output filtered video
 */
template <class T, class T2>
void VideoProcessing::medfilt2(const Video<T> &input, Video<T2> &output,vector<int> sz){
    output.reset();
    if(sz.size()!=2){
        printf("filter size should be 2 in medfilt2\n");
    }
    vector<T> buffer(sz[0]*sz[1]);
    vector<int> dims = input.dimensions();
    vector<int> mins(2);
    vector<int> maxs(2);
    for (size_t i = 0; i < sz.size(); ++i) {
        mins[i] = min(sz[i]/2,dims[i]-sz[i]/2);
        maxs[i] = max(dims[i]-sz[i]/2,sz[i]/2);
    }

    const T* pInput = input.dataReader();
    T* pOutput = output.dataWriter();
    for (int k = 0; k < dims[2]; ++k)
        for (int j = 0; j < dims[1]; ++j)
            for (int i = 0; i < dims[0]; ++i)
    {
        for (int chan = 0; chan < dims[3]; ++chan) {
            fill(buffer.begin(), buffer.end(),0);
            for (int f0 = 0; f0 < sz[0]; ++f0) 
                for (int f1 = 0; f1 < sz[0]; ++f1)
            {
                int I = (i-sz[0]/2+f0);
                int J = (j-sz[1]/2+f1);
                if( I < 0 ) {
                    I *= -1;
                }
                if( J < 0 ) {
                    J *= -1;
                }
                if(I > dims[0] - 1){
                    I = 2*(dims[0]-1) - I;
                }
                if(J > dims[1] - 1){
                    J = 2*(dims[1]-1) - J;
                }
                int index =  I + dims[0]*( J + dims[1]*(k+dims[2]*chan));
                buffer[f0+sz[0]*f1] = pInput[index];
            }
            sort(buffer.begin(), buffer.end());
            pOutput[i+dims[0]*(j+dims[1]*(k+dims[2]*chan))] = buffer[(sz[0]*sz[1])/2];
        }
    }
}

template <class T1, class T2>
void VideoProcessing::backwardWarp(Video<T1> &source, const Video<T2> warpField) {
    Video<T1> target = Video<T1>(source.size());
    backwardWarp(source,warpField,target);
    source.copy(target);
}

template <class T1, class T2>
void VideoProcessing::backwardWarp(const Video<T1> &source, const Video<T2> warpField, Video<T1> &target) {
    // TODO: check dims
    target.reset();
    vector<int> dims = target.dimensions();
    int nVoxels = target.voxelCount();
    T1* pTarget = target.dataWriter();
    double queryPoint[3];
    int index;
    T1* interpolated = new T1[dims[3]];
    const T2* pWarp = warpField.dataReader();
    for (int k = 0; k < dims[2]; ++k)
        for (int j = 0; j < dims[1]; ++j)
            for (int i = 0; i < dims[0]; ++i)
    {
        index = i +dims[0]*(j +dims[1]*k);
        queryPoint[0] = i + pWarp[index+1*nVoxels]; // y
        queryPoint[1] = j + pWarp[index+0*nVoxels]; // x
        queryPoint[2] = k + pWarp[index+2*nVoxels]; // t
        bool inside = (queryPoint[0]>=0) && (queryPoint[0]<dims[0]);
        inside = inside && (queryPoint[1]>=0) && (queryPoint[1]<dims[1]);
        inside = inside && (queryPoint[2]>=0) && (queryPoint[2]<dims[2]);
        if( inside ) {
            trilinearInterpolate(source, queryPoint, interpolated);
            for (int chan = 0; chan < dims[3]; ++chan) {
                pTarget[index + chan*nVoxels] = interpolated[chan];
            }
        }else {
            for (int chan = 0; chan < dims[3]; ++chan) {
                pTarget[index + chan*nVoxels] = 0;
            }
        }
    }
    delete interpolated;
}

template <class T1, class T2>
void VideoProcessing::backwardWarp2D(const Video<T1> &source, const Video<T2> warpField, Video<T1> &target, int frameOffset) {
    // TODO: check dims
    target.reset();
    vector<int> dims = target.dimensions();
    int nVoxels = target.voxelCount();
    T1* pTarget = target.dataWriter();
    double queryPoint[3];
    int index;
    T1* interpolated = new T1[dims[3]];
    const T2* pWarp = warpField.dataReader();
    for (int k = 0; k < dims[2]; ++k)
        for (int j = 0; j < dims[1]; ++j)
            for (int i = 0; i < dims[0]; ++i)
    {
        index = i +dims[0]*(j +dims[1]*k);
        queryPoint[0] = i + pWarp[index+1*nVoxels]; // y
        queryPoint[1] = j + pWarp[index+0*nVoxels]; // x
        queryPoint[2] = k + frameOffset;// t
        bool inside = (queryPoint[0]>=0) && (queryPoint[0]<dims[0]);
        inside = inside && (queryPoint[1]>=0) && (queryPoint[1]<dims[1]);
        inside = inside && (queryPoint[2]>=0) && (queryPoint[2]<dims[2]);
        if( inside ) {
            trilinearInterpolate(source, queryPoint, interpolated);
            for (int chan = 0; chan < dims[3]; ++chan) {
                pTarget[index + chan*nVoxels] = interpolated[chan];
            }
        }else {
            for (int chan = 0; chan < dims[3]; ++chan) {
                pTarget[index + chan*nVoxels] = 0;
            }
        }
    }
    delete interpolated;
}

template <class T, class T2>
void VideoProcessing::dx(const Video<T> &input, Video<T2> &dest, bool advancedFilter) {
    dest.reset();
    if( advancedFilter ) {
		double xFilter[5]={1,-8,0,8,-1};
		for(int i=0;i<5;i++){
			xFilter[i]/=12;
        }
		hfiltering(input,dest,xFilter,2);
    }else {
        const T* pData = input.dataReader();
        T2* pDest = dest.dataWriter();
        vector<int> dims = input.dimensions();
        int nVoxels = input.voxelCount();
        int index;
		for(int k=0;k<dims[2];k++)
			for(int j=0;j<dims[1]-1;j++)
                for(int i=0;i<dims[0];i++)
			{
				index = i+dims[0]*(j+dims[1]*k);
				for(int l=0;l<dims[3];l++)
					pDest[index+nVoxels*l] = static_cast<T2>(pData[(index+dims[0])+nVoxels*l])-static_cast<T2>(pData[index+nVoxels*l]);
			}
    }
}

template <class T, class T2>
void VideoProcessing::dy(const Video<T> &input, Video<T2> &dest, bool advancedFilter) {
    dest.reset();
    if( advancedFilter ) {
		double xFilter[5]={1,-8,0,8,-1};
		for(int i=0;i<5;i++){
			xFilter[i]/=12;
        }
		vfiltering(input,dest,xFilter,2);
    }else {
        const T* pData = input.dataReader();
        T2* pDest = dest.dataWriter();
        vector<int> dims = input.dimensions();
        int nVoxels = input.voxelCount();
        int index;
		for(int k=0;k<dims[2];k++)
			for(int j=0;j<dims[1];j++)
                for(int i=0;i<dims[0]-1;i++)
			{
				index = i+dims[0]*(j+dims[1]*k);
				for(int l=0;l<dims[3];l++)
					pDest[index+nVoxels*l] = static_cast<T2>(pData[(index+1)+nVoxels*l])-static_cast<T2>(pData[index+nVoxels*l]);
			}
    }
}

template <class T, class T2>
void VideoProcessing::dt(const Video<T> &input, Video<T2> &dest, bool advancedFilter) {
    dest.reset();
    if( advancedFilter ) {
		double xFilter[5]={1,-8,0,8,-1};
		for(int i=0;i<5;i++){
			xFilter[i]/=12;
        }
		tfiltering(input,dest,xFilter,2);
    }else {
        const T* pData = input.dataReader();
        T2* pDest = dest.dataWriter();
        vector<int> dims = input.dimensions();
        int nVoxels = input.voxelCount();
        int index;
		for(int k=0;k<dims[2]-1;k++)
			for(int j=0;j<dims[1];j++)
                for(int i=0;i<dims[0];i++)
			{
				index = i+dims[0]*(j+dims[1]*k);
				for(int l=0;l<dims[3];l++)
					pDest[index+nVoxels*l] = static_cast<T2>(pData[(index+dims[0]*dims[1])+nVoxels*l])-static_cast<T2>(pData[index+nVoxels*l]);
			}
    }
}

template <class T1, class T2, class T3>
void VideoProcessing::forwardWarp(  const Video<T1> &input,
                                    const Video<T2> &warpField,
                                    const Video<T2> &cost,
                                    int splatSz[3],
                                    Video<T1> &output,
                                    Video<T3> &mask)
{
    int splatX = splatSz[0];
    int splatY = splatSz[1];
    int splatT = splatSz[2];
    int nSplat = splatX*splatY*splatT;

    // TODO: exception
    if( (splatX%2 != 1) || (splatY%2 != 1)|| (splatT%2 != 1) ) {
        fprintf(stderr, "Splat size must be odd.\n");
    }

    VideoSize sz  = warpField.size();
    int width     = sz.width;
    int height    = sz.height;
    int nFrames   = sz.nFrames;
    int nChannels = sz.nChannels;
    int nVoxels   = warpField.voxelCount();

    Video<T2> targetCost(height,width,nFrames,1);
    T2* pTargetCost = targetCost.dataWriter();
    targetCost.reset(cost.max()+1);
    
    const T2* pWarpField = warpField.dataReader();
    const T1* pInput     = input.dataReader();
    const T2* pCost      = cost.dataReader();
    T1* pOut             = output.dataWriter();
    T3* pMask            = mask.dataWriter();

    for(int k=0;k<nFrames;++k)
        for(int j=0;j<width;++j)
            for(int i=0;i<height;++i)
    {
        int index = i + height*( j + width*k );

        // Compute center target positions
        vector<vector<T2> > newPos(3);
        vector<bool> inside(nSplat,true);
        for(int wchan = 0; wchan<3; wchan++) {
            int indexW         = index + height*width*nFrames*wchan;
            T2 offset = pWarpField[indexW];
            switch(wchan) {
                case 0:
                    offset += j ;
                    break;
                case 1:
                    offset += i ;
                    break;
                case 2:
                    offset += k ;
                    break;
            }
            newPos[wchan] = vector<T2>(nSplat,offset);
        }
        // Compute offset for splatted voxels;
        for(int kN = 0; kN<splatT ;kN++) 
            for(int jN = 0; jN<splatX ;jN++) 
                for(int iN = 0; iN<splatY ;iN++)
        {
            int neighbor = iN + splatY*(jN + splatT*kN);
            newPos[0][neighbor] += jN - (splatX-1)/2;
            newPos[1][neighbor] += iN - (splatY-1)/2;
            newPos[2][neighbor] += kN - (splatT-1)/2;
        }

        // Check which positions are inside the rendering area
        for( int neighbor = 0; neighbor < nSplat ; neighbor++) {
            T2 p = newPos[0][neighbor];
            if( p < 0 || p > width-1 )  { inside[neighbor] = false; }
            p = newPos[1][neighbor];
            if( p < 0 || p > height-1 ) { inside[neighbor] = false; }
            p = newPos[2][neighbor];
            if( p < 0 || p > nFrames-1 ) { inside[neighbor] = false; }
        }

        T2 c = pCost[index];
        for( int neighbor = 0; neighbor < nSplat; neighbor++) {
            if( !inside[neighbor] ) {
                // Don't process outside voxels
                continue;
            }
            int indexTarget = round(newPos[1][neighbor])
                            + height*( round(newPos[0][neighbor]) 
                            + width*round(newPos[2][neighbor]) );
            if( c > pTargetCost[indexTarget]){
                // Dont update this voxel, it's warping field is less reliable
                // than the current one
                continue;
            }
            if(pMask[indexTarget] == 0){
                pMask[indexTarget] = 127;
            }else{
                pMask[indexTarget] += 1;
            }
            for(int chan = 0; chan<nChannels; chan++) {
                pOut[indexTarget+nVoxels*chan] = pInput[index+nVoxels*chan];
                // compensate for roundoff
                pOut[indexTarget+nVoxels*chan] += 
                    round(newPos[chan][neighbor])-newPos[chan][neighbor];
                pTargetCost[indexTarget] = c;
            }
        }
    }
}

template <class T1>
void VideoProcessing::videoFuse( const Video<T1> &inputA, const Video<T1> &inputB, Video<T1> &output) {
    output.reset();
    // TODO: check output has 3channels
    int nVoxelsA = inputA.voxelCount();
    int nVoxelsB = inputB.voxelCount();
    int nChannels = inputA.channelCount();
    int height = inputA.getHeight();
    int width = inputA.getWidth();
    int nFramesA = inputA.frameCount();
    int nFramesB = inputB.frameCount();

    output = IVideo(height,width,max(nFramesA,nFramesB),nChannels);
    int nVoxels = output.voxelCount();

    T1* pOutput = output.dataWriter();
    const T1* pA = inputA.dataReader();
    const T1* pB = inputB.dataReader();
    double buffer[2];
    for(int k=0;k<max(nFramesA,nFramesB); ++k)
        for(int j=0;j<width; ++j)
            for(int i = 0; i<height ; ++i)
    {
        int index = i + height*( j + width*k );
        buffer[0] = 0;
        buffer[1] = 0;
        for (int chan = 0; chan < nChannels; ++chan) {
            if(k<nFramesA){
                buffer[0] += pA[index+chan*nVoxelsA];
            }
            if(k<nFramesB){
                buffer[1] += pB[index+chan*nVoxelsB];
            } 
        }
        buffer[0] /= nChannels;
        buffer[1] /= nChannels;

        pOutput[index] = buffer[0];
        pOutput[index+2*nVoxels] = buffer[0];
        pOutput[index+nVoxels] = buffer[1];
    }
}

template <class T>
void VideoProcessing::dilate3D(Video<T> &video, int* fsize) {
    vector<int> dims = video.dimensions();
    if(dims[3] > 1) {
        // TODO: error
        cout << "Warning: dilatation should be used on binary video\n" << endl;
    }
    Video<T> cp(video);

    const T* pV = video.dataReader();
    T* pCopy = cp.dataWriter();
    for (int k = 0; k < dims[2]; ++k)
        for (int j = 0; j < dims[1]; ++j)
            for (int i = 0; i < dims[0]; ++i)
    {
        int index = i + dims[0]*( j + dims[1]*k );
        if( pV[index] == 0 ){
            continue;
        }
        for (int k_w = max(k-fsize[2],0); k_w <= min(k+fsize[2],dims[2]-1); ++k_w)
            for (int  j_w = max(j-fsize[1],0); j_w <= min(j+fsize[1],dims[1]-1); ++j_w)
                for (int i_w = max(i-fsize[0],0); i_w <= min(i+fsize[0],dims[0]-1); ++i_w)
        {
            int index_w = i_w + dims[0]*( j_w + dims[1]*k_w );
            pCopy[index_w] = pV[index];
        }
    }

    video.copy(cp);
}

template <class T, class T2>
void VideoProcessing::edges3D(const Video<T> &input, Video<T2> &edgemap, int channelStart, int channelEnd, float thresh) {
    double smoothFilter[3] = {1, 2, 1};
    for (int i = 0; i < 3; ++i) {
        smoothFilter[i] /= 4;
    }
    double derivFilter[3] = {-1, 0, 1};
    int fsize = 1;

    // TODO: uncomment the time part?

    Video<T> src = input.extractChannel(channelStart, channelEnd);
    src.collapse();
    Video<float> buffer(src.size());
    Video<float> buffer2(src.size());
    Video<float> gx(src.size());
    Video<float> gy(src.size());
    // Video<float> gt(src.size());

    hfiltering(src, buffer, derivFilter, fsize);
    vfiltering(buffer, buffer2, smoothFilter, fsize);
    // tfiltering(buffer2, gx, smoothFilter, fsize);

    vfiltering(src, buffer, derivFilter, fsize);
    hfiltering(buffer, buffer2, smoothFilter, fsize);
    // tfiltering(buffer2, gy, smoothFilter, fsize);

    // tfiltering(src, buffer, derivFilter, fsize);
    // hfiltering(buffer, buffer2, smoothFilter, fsize);
    // vfiltering(buffer2, gt, smoothFilter, fsize);

    edgemap             = Video<T2>(gx.size());
    T2 *pOut = edgemap.dataWriter();
    const float* pX         = gx.dataReader();
    const float* pY         = gy.dataReader();
    // const float* pT         = gt.dataReader();
    thresh *= thresh;
    for (int i = 0; i < edgemap.voxelCount(); ++i) {
        float val = pX[i]*pX[i] + pY[i]*pY[i];
        // float val = pX[i]*pX[i] + pY[i]*pY[i] +pT[i]*pT[i];
        val /= 3;
        if(val > thresh){
            pOut[i] = 255;
        }
    }
}

#endif /* end of include guard: VIDEOPROCESSING_HPP_JSKF1JAR */
