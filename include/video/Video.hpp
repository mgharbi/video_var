/* --------------------------------------------------------------------------
 * File:    Video.hpp
 * Author:  Michael Gharbi <gharbi@mit.edu>
 * Created: 2014-01-21
 * --------------------------------------------------------------------------
 * 
 * Basic Video data class.
 * 
 * ------------------------------------------------------------------------*/


#ifndef VIDEO_HPP_8XGAVAGK
#define VIDEO_HPP_8XGAVAGK

#include <iostream>
#include <typeinfo>

#include <opencv2/opencv.hpp>
#include <boost/filesystem.hpp>
#include <boost/format.hpp>

#include "video/VideoExceptions.hpp"

using namespace std;
using namespace cv;
namespace fs = boost::filesystem;

/** 
 * Video dimensions.
 */
typedef struct VideoSize{
    int height, width, nFrames, nChannels;
    VideoSize(int h, int w, int nF, int nC) : 
        height(h), width(w), nFrames(nF), nChannels(nC) {};
} VideoSize;

/** 
 * 3D array that represents a video volume and implements basic video 
 * operations.
 *
 */
template <class T>
class Video
{
public:
    Video();
    Video(fs::path path);
    Video(int height, int width, int nFrames, int nChannels);
    Video(VideoSize s);
    Video(const Video& source);

    virtual ~Video();

    virtual void load(fs::path path);
    void exportFrame(int frame,fs::path path) const;
    void exportXTslice(int y,fs::path path) const;
    void exportYTslice(int x,fs::path path) const;
    void exportScaledFrame(int frame,fs::path path, T min, T max) const;
    void exportVideo(fs::path path,string name) const;
    void exportXTvideo(fs::path path,string name) const;
    void exportYTvideo(fs::path path,string name) const;

    void reset(T value = (T) 0);
    Video<T>& operator=(const Video<T>& source);

    inline VideoSize size() const;
    inline vector<int> dimensions() const;
    inline int elementCount() const;
    inline int voxelCount() const;
    inline int channelCount() const {return nChannels;};
    inline int frameCount() const {return nFrames;};
    inline int getHeight() const {return height;};
    inline int getWidth() const {return width;};

    inline T max(int channel=-1) const;
    inline T min(int channel=-1) const;
    inline T mean(int channel=-1) const;
    inline T meanFrame(int frame, int channel=-1) const;
    inline void clamp(T min, T max);
    void scalarMultiply(T value);
    void scalarAdd(T value);
    void scalarMultiplyChannel(T value, int chan);
    template <class T2> void multiply(Video<T2> other) 
        throw(IncorrectSizeException) ;
    template <class T2> void add(Video<T2> other) 
        throw(IncorrectSizeException) ;
    template <class T2> void subtract(Video<T2> other) 
        throw(IncorrectSizeException) ;

    inline const T* dataReader() const {return pData;};
    inline const T* channelReader(int k) const {
        return pData+k*height*width*nFrames; };
    inline T* dataWriter() {return pData;};
    inline T* channelWriter(int k) const {
        return pData+k*height*width*nFrames; };
    inline T at(int i, int j, int k, int l) const { 
        return pData[i + height*(j + width*(k + nFrames*l))]; };
    inline void setAt(T val, int i, int j, int k, int l) { 
        pData[i + height*(j + width*(k + nFrames*l))] = val; };

    template <class T2> void copy(const Video<T2>& source);
    Video<T> extractChannel(int start, int end=-1) const;
    void addChannels(int chanCount);
    template <class T2>
    void writeChannel(int chan, const Video<T2> &source);

    void collapse();

    // MATLAB I/O
    void initFromMxArray(int ndims, int* dims, const T* data);
    void copyToMxArray(int n, T* data) const;

protected:
    T* pData;
    int height, width, nFrames, nChannels;
    int nVoxels, nElements;

    virtual void allocate(int h, int w, int nF, int nC);
    virtual void clear();
    template <class T2> bool checkSizeMatch(const Video<T2> &other);
};


#pragma mark - inlined methods

/**
 * Return the video dimensions wrapped in a VideoSize struct.
 * @param[out] sz the video size struct.
 */
template <class T>
VideoSize Video<T>::size() const{
    return VideoSize(height,width,nFrames,nChannels);
}

/**
 * Return the video dimensions wrapped in a vector.
 * @param[out] dims 4D vector with the dimensions
 */
template <class T>
vector<int> Video<T>::dimensions() const{
    vector<int> dims(4);
    dims[0] = height;
    dims[1] = width;
    dims[2] = nFrames;
    dims[3] = nChannels;
    return dims;
}

/**
 * Number of elements in the underlying array
 */
template <class T>
int Video<T>::elementCount() const {
    return nElements;
}

/**
 * Number of elements per channel in the underlying array
 */
template <class T>
int Video<T>::voxelCount() const {
    return nVoxels;
}

/**
 * Max element of the array
 * @param[in] channel (optional) limit the search to this channel
 */
template <class T>
T Video<T>::max(int channel) const {
    T m = pData[0];
    if(channel < 0) {
        for (int i = 0; i < nElements; ++i) {
            m = (pData[i]>m) ? pData[i]: m;
        }
    }else {
        for (int i = 0; i < nVoxels; ++i) {
            m = (pData[i+channel*nVoxels]>m) ? pData[i+channel*nVoxels]: m;
        }
    }
    return m;
}

/**
 * Mean of the array
 * @param[in] channel (optional) limit the search to this channel
 */
template <class T>
T Video<T>::mean(int channel) const {
    double m= 0;
    if(channel < 0) {
        for (int i = 0; i < nElements; ++i) {
            m += pData[i];
        }
        m /= nElements;
    }else {
        for (int i = 0; i < nVoxels; ++i) {
            m += pData[i+channel*nVoxels];
        }
        m /= nVoxels;
    }
    return m;
}

/**
 * Mean of the asked frame
 * @param[in] channel (optional) limit the search to this channel
 */
template <class T>
T Video<T>::meanFrame(int frame, int channel) const {
    double m = 0;
    if(channel < 0) {
        for(int k=0;k<nChannels;k++)
            for(int j=0;j<width;j++)
                for(int i=0;i<height;i++)
        {
            int index = i+height*(j+width*frame) + k*nVoxels;
            m += pData[index];
        }
        m /= height*width*nChannels;
    }else {
        for(int j=0;j<width;j++)
            for(int i=0;i<height;i++)
        {
            int index = i+height*(j+width*frame) + channel*nVoxels;
            m += pData[index];
        }
        m /= height*width;
    }
    return m;
}

/**
 * Min element of the array
 * @param[in] channel (optional) limit the search to this channel
 */
template <class T>
T Video<T>::min(int channel) const {
    T m = pData[0];
    if(channel < 0) {
        for (int i = 0; i < nElements; ++i) {
            m = (pData[i]<m) ? pData[i]: m;
        }
    }else {
        for (int i = 0; i < nVoxels; ++i) {
            m = (pData[i+channel*nVoxels]<m) ? pData[i+channel*nVoxels]: m;
        }
    }
    return m;
}

#pragma mark - templated methods

/**
 * Check the dimensions of the two videos match.
 */
template <class T>
template <class T2>
bool Video<T>::checkSizeMatch(const Video<T2> &other) {
    vector<int> dims = other.dimensions();
    vector<int> dims2 = this->dimensions();
    for (size_t i = 0; i < dims.size(); ++i) {
        if(dims[i] != dims2[i]){
            return false;
        }
    }
    return true;
}

/**
 * Copy the content of the source video, thus erasing the current content.
 */
template <class T>
template <class T2>
void Video<T>::copy(const Video<T2>& source) {
    clear();
    vector<int> dims = source.dimensions();
    allocate(dims[0], dims[1], dims[2], dims[3]);
    if(typeid(T)==typeid(T2)) {
        memcpy(pData, source.dataReader(), nElements*sizeof(T));
    }else{
        const T2* pSource = source.dataReader();
        for (int i = 0; i < nElements; ++i) {
            pData[i] = static_cast<T>(pSource[i]);
        }
    }
}

/**
 * Element-wise multiply in place with the other array.
 */
template <class T>
template <class T2>
void Video<T>::multiply(Video<T2> other) throw(IncorrectSizeException) {
    if(!checkSizeMatch(other)) throw IncorrectSizeException();
    const T2* pOther = other.dataReader();
    if(pData != NULL) {
        for (int i = 0; i < nElements; ++i) {
            pData[i] *= (T) pOther[i];
        }
    }
}

/**
 * Element-wise addition in place with the other array.
 */
template <class T>
template <class T2>
void Video<T>::add(Video<T2> other) throw(IncorrectSizeException){
    if(!checkSizeMatch(other)) throw IncorrectSizeException();
    const T2* pOther = other.dataReader();
    if(pData != NULL) {
        for (int i = 0; i < nElements; ++i) {
            pData[i] += (T) pOther[i];
        }
    }
}

/**
 * Element-wise subtraction in place with the other array.
 */
template <class T>
template <class T2>
void Video<T>::subtract(Video<T2> other) throw(IncorrectSizeException){
    if(!checkSizeMatch(other)) throw IncorrectSizeException();
    const T2* pOther = other.dataReader();
    if(pData != NULL) {
        for (int i = 0; i < nElements; ++i) {
            pData[i] -= (T) pOther[i];
        }
    }
}

template <class T>
template <class T2>
void Video<T>::writeChannel(int chan, const Video<T2> &source) {
   T* pW =  channelWriter(chan);
   const T2* pR =  source.dataReader();
   for (int i = 0; i < nVoxels; ++i) {
       pW[i] = (T) pR[i];
   }
}

template <class T>
void Video<T>::clamp(T min, T max) {
    for (int i = 0; i < nElements; ++i) {
        if(pData[i]<min) {
            pData[i] = min;
        }
        if(pData[i]>max) {
            pData[i] = max;
        }
    }
}

#pragma mark - type aliases

typedef Video<unsigned char> IVideo;
typedef Video<float> FVideo;
typedef Video<double> DVideo;


#endif /* end of include guard: VIDEO_HPP_8XGAVAGK */

