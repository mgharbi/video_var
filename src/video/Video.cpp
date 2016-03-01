/* --------------------------------------------------------------------------
 * File:    Video.cpp
 * Author:  Michael Gharbi <gharbi@mit.edu>
 * Created: 2014-01-29
 * --------------------------------------------------------------------------
 * 
 * Basic Video data class.
 * 
 * ------------------------------------------------------------------------*/


#include "video/Video.hpp"
#include <cassert>


/** 
 * Instantiate an empty video.
 */
template <class T>
Video<T>::Video() {
    allocate(0,0,0,0);
}

/** 
 * Instantiate a video with all zeros of the given size.
 * @param[in] h height of the video.
 * @param[in] w width of the video.
 * @param[in] nF number of frames.
 * @param[in] nC number of channels.
 */
template <class T>
Video<T>::Video(int h, int w, int nF, int nC) {
    allocate(h, w, nF, nC);
}

/** 
 * Instantiate a video with all zeros of the given size.
 * @param[in] s size.
 */
template <class T>
Video<T>::Video(VideoSize s) {
    allocate(s.height, s.width, s.nFrames, s.nChannels);
}

/** 
 * Copy constructor.
 * @param[in] source other video to copy from.
 */
template <class T>
Video<T>::Video(const Video<T>& source) {
    allocate(0,0,0,0);
    copy(source);
}

/** 
 * Move constructor.
 * @param[in] source other video to copy from.
 */
template <class T>
Video<T>::Video(Video<T>&& source) :
    pData(source.pData),
    height(source.height),
    width(source.width),
    nFrames(source.nFrames),
    nChannels(source.nChannels),
    nVoxels(source.nVoxels),
    nElements(source.nElements)
{
    source.pData = NULL;
}


template <class T>
Video<T>::~Video() {
    if( pData != NULL ) {
        delete[] pData;
        pData = NULL;
    }
}

/** 
 * Convert from MATLAB data
 * @param[in] ndims
 * @param[in] dims
 * @param[in] data
 */
template <class T>
void Video<T>::initFromMxArray(int ndims, int* dims, const T* data) {
    assert(ndims == 4);
    clear();
    allocate(dims[0],dims[1],dims[2],dims[3]);
    for (int x = 0; x < dims[0]*dims[1]*dims[2]*dims[3]; ++x) {
        pData[x] = data[x];
    }

}

template <class T>
void Video<T>::copyToMxArray(int n, T* data) const {
    assert(n == height*width*nFrames*nChannels);
    memcpy(data, pData, n*sizeof(T));
    // for (int x = 0; x < height*width*nFrames*nChannels; ++x) {
    //     data[x] = pData[x];
    // }

}

template <class T>
void Video<T>::addChannels(int chanCount) {
    Video<T> temp(*this);
    allocate(height, width, nFrames, nChannels+chanCount);
    const T * pTemp = temp.dataReader();
    for (int i = 0; i < temp.elementCount(); ++i) {
        pData[i] = pTemp[i];
    }
}




template <class T>
void Video<T>::allocate(int h, int w, int nF, int nC) {
    height    = h;
    width     = w;
    nFrames   = nF;
    nChannels = nC;
    nVoxels   = height*width*nFrames;
    nElements = nVoxels*nChannels;
    pData     = NULL;

    if(nElements > 0) {
        pData = new T[nElements];
        if(pData==NULL) {
            fprintf(stderr,"Error in allocating a video.\n");
        }
		memset(pData,0,sizeof(T)*nElements);
    }
}

template <class T>
Video<T>& Video<T>::operator=(const Video<T>& source) {
    if(this != &source) {
        copy(source);
    }
    return *this;
}

template <class T>
Video<T>& Video<T>::operator=(Video<T>&& source) {
    if(this != &source) {
        clear();
        pData     = source.pData;
        height    = source.height;
        width     = source.width;
        nFrames   = source.nFrames;
        nChannels = source.nChannels;
        nVoxels   = source.nVoxels;
        nElements = source.nElements;
        source.pData = NULL;
    }
	return *this;
}

/** 
 * Clear all data from the video container.
 */
template <class T>
void Video<T>::clear() {
    if( pData != NULL ) {
        delete[] pData;
        pData     = NULL;
    }
    height    = 0;
    width     = 0;
    nFrames   = 0;
    nChannels = 0;
    nVoxels   = 0;
    nElements = 0;
}

/** 
 * Reset the video container.
 * @param[in] value value to reset the video to.
 */
template <class T>
void Video<T>::reset(T value) {
    if(pData != NULL) {
        for(int i = 0; i<nElements; i++) {
            pData[i] = value;
        }
    }
}


template <class T>
Video<T> Video<T>::extractChannel(int start, int end) const {
    int nChan = 1;
    if(end>0) {
        nChan = end-start+1;
    }
    Video<T> chan = Video<T>(height,width,nFrames,nChan);
    for (int i = 0; i < nChan; ++i) {
        const T* chanReader = channelReader(i+start);
        memcpy(chan.channelWriter(i),chanReader,sizeof(T)*nVoxels);
    }
    return chan;
}

template <class T>
void Video<T>::scalarMultiply(T value) {
    if(pData != NULL) {
        for (int i = 0; i < nElements; ++i) {
            pData[i] *= value;
        }
    }
}
template <class T>
void Video<T>::scalarAdd(T value) {
    if(pData != NULL) {
        for (int i = 0; i < nElements; ++i) {
            pData[i] += value;
        }
    }
}

template <class T>
void Video<T>::scalarMultiplyChannel(T value, int chan) {
    if(pData != NULL) {
        for (int i = 0; i < nVoxels; ++i) {
            pData[i + chan*nVoxels] *= value;
        }
    }
}

template <class T>
void Video<T>::collapse() {
    if( nChannels < 2){
        // No need to collapse any channel
        return;
    }
    Video<T> temp(height,width,nFrames,1);
    T* pTemp = temp.dataWriter();
    double buffer;
    for (int i = 0; i < nVoxels; ++i) {
        buffer = 0;
        for (int k = 0; k < nChannels; ++k) {
            buffer += pData[i+k*nVoxels];
        }
        buffer /= nChannels;
        pTemp[i] = buffer;
    }
    this->copy(temp);
}

template <class T>
Video<T>& Video<T>::operator+=(const Video<T> &other) {
    assert(elementCount() == other.elementCount());
    for (int i = 0; i < elementCount(); ++i) {
        at(i) += other.at(i);
    }
    return *this;
}
template <class T>
Video<T>& Video<T>::operator-=(const Video<T> &other) {
    assert(elementCount() == other.elementCount());
    for (int i = 0; i < elementCount(); ++i) {
        at(i) -= other.at(i);
    }
    return *this;
}
template <class T>
Video<T>& Video<T>::operator*=(const Video<T> &other) {
    assert(elementCount() == other.elementCount());
    for (int i = 0; i < elementCount(); ++i) {
        at(i) *= other.at(i);
    }
    return *this;
}
template <class T>
Video<T>& Video<T>::operator/=(const Video<T> &other) {
    assert(elementCount() == other.elementCount());
    for (int i = 0; i < elementCount(); ++i) {
        at(i) /= other.at(i);
    }
    return *this;
}

template <class T>
Video<T> Video<T>::operator+(const Video<T> &other) {
    return Video<T>(*this) += other;
}
template <class T>
Video<T> Video<T>::operator-(const Video<T> &other) {
    return Video<T>(*this) -= other;
}
template <class T>
Video<T> Video<T>::operator*(const Video<T> &other) {
    return Video<T>(*this) *= other;
}
template <class T>
Video<T> Video<T>::operator/(const Video<T> &other) {
    return Video<T>(*this) /= other;
}



template class Video<unsigned char>;
template class Video<int>;
template class Video<float>;
template class Video<double>;
