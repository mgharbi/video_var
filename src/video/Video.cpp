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
#include <boost/format.hpp>
#include <cassert>

#pragma mark - lifecycle

/** 
 * Instantiate a video from file.
 * @param[in] path path of the video to load.
Elements*/
template <class T>
Video<T>::Video(fs::path path) {
    load(path);
}

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
    for (int x = 0; x < height*width*nFrames*nChannels; ++x) {
        data[x] = pData[x];
    }

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
	copy(source);
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

#pragma mark - common operations

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


#pragma mark - io

/** 
 * Load a RGB video into the current container.
 * @param[in] path path to the input video.
 */
template <class T>
void Video<T>::load(fs::path path) {
    VideoCapture vc(path.string());
    if (!vc.isOpened()) {
        fprintf(stderr,"Error in loading video %s.\n",path.c_str());
        return;
    }

    int h  = static_cast<int>( vc.get(CV_CAP_PROP_FRAME_HEIGHT) );
    int w  = static_cast<int>( vc.get(CV_CAP_PROP_FRAME_WIDTH) );
    int nF = static_cast<int>( vc.get(CV_CAP_PROP_FRAME_COUNT) );
    allocate(h,w,nF,3);
    // TODO: check channels
    
    Mat frame;
    int index = 0;
    Vec<uchar,3> elt;
    for(int k = 0; k< nFrames; ++k) {
        vc >> frame;
        for (int j = 0; j < width; ++j)
            for (int i = 0; i < height; ++i)
        {
            index = i + height*(j+width*k);
            elt = frame.at<Vec<uchar,3> >(i,j);
            pData[index] = elt[0];
            pData[index+nVoxels] = elt[1];
            pData[index+2*nVoxels] = elt[2];
        }
    }
    vc.release();
}

/** 
 * Export a frame to .jpg .
 * @param[in] frame frame number.
 * @param[in] path output path.
 */
template <class T>
void Video<T>::exportFrame(int frame,fs::path path) const {
    if(typeid(T) != typeid(unsigned char)) {
        fprintf(stderr,"Video must be of type unsigned char to export.\n");
        return;
    }
    if(nChannels == 3) {
        Mat out(height,width,CV_8UC3);
        const T* data = dataReader();
        Vec<uchar,3> elt;
        int index;
        for (int j = 0; j < width; ++j)
            for (int i = 0; i < height; ++i) {
                index = i + height*(j+width*frame);
                for(int l = 0; l<nChannels ; l++) {
                    elt[l] = data[index+l*nVoxels];
                }
                out.at<Vec<uchar,3> >(i,j) = elt;
            }
        imwrite(path.string(),out);
    } else if(nChannels == 1) {
        Mat out(height,width,CV_8UC3);
        const T* data = dataReader();
        Vec<uchar,3> elt;
        int index;
        for (int j = 0; j < width; ++j)
            for (int i = 0; i < height; ++i) {
                index = i + height*(j+width*frame);
                for(int l = 0; l<3 ; l++) {
                    elt[l] = data[index];
                }
                out.at<Vec<uchar,3> >(i,j) = elt;
            }
        imwrite(path.string(),out);
    } else {
        fprintf(stderr,"Video must have 1 or 3 channels.\n");
        return;
    }
}

/** 
 * Export a XT slice to .jpg .
 * @param[in] frame frame number.
 * @param[in] path output path.
 * time goes from bottom to top
 */
template <class T>
void Video<T>::exportXTslice(int y,fs::path path) const {
    if(typeid(T) != typeid(unsigned char)) {
        fprintf(stderr,"Video must be of type unsigned char to export.\n");
        return;
    }
    if(nChannels == 3) {
        Mat out(nFrames,width,CV_8UC3);
        const T* data = dataReader();
        Vec<uchar,3> elt;
        int index;
        for (int k = 0; k < nFrames; ++k) 
            for (int j = 0; j < width; ++j){
                index = y + height*(j+width*k);
                for(int l = 0; l<nChannels ; l++) {
                    elt[l] = data[index+l*nVoxels];
                }
                out.at<Vec<uchar,3> >(nFrames-k-1,j) = elt;
            }
        imwrite(path.string(),out);
    } else if(nChannels == 1) {
        Mat out(height,width,CV_8UC3);
        const T* data = dataReader();
        Vec<uchar,3> elt;
        int index;
        for (int k = 0; k < nFrames; ++k) 
            for (int j = 0; j < width; ++j){
                index = y + height*(j+width*k);
                for(int l = 0; l<3 ; l++) {
                    elt[l] = data[index];
                }
                out.at<Vec<uchar,3> >(nFrames-k-1,j) = elt;
            }
        imwrite(path.string(),out);
    } else {
        fprintf(stderr,"Video must have 1 or 3 channels.\n");
        return;
    }
}

/** 
 * Export a YT slice to .jpg .
 * @param[in] frame frame number.
 * @param[in] path output path.
 */
template <class T>
void Video<T>::exportYTslice(int x,fs::path path) const {
    if(typeid(T) != typeid(unsigned char)) {
        fprintf(stderr,"Video must be of type unsigned char to export.\n");
        return;
    }
    if(nChannels == 3) {
        Mat out(height,nFrames,CV_8UC3);
        const T* data = dataReader();
        Vec<uchar,3> elt;
        int index;
        for (int k = 0; k < nFrames; ++k) 
            for (int i = 0; i < height; ++i) {
                index = i + height*(x+width*k);
                for(int l = 0; l<nChannels ; l++) {
                    elt[l] = data[index+l*nVoxels];
                }
                out.at<Vec<uchar,3> >(i,k) = elt;
            }
        imwrite(path.string(),out);
    } else if(nChannels == 1) {
        Mat out(height,width,CV_8UC3);
        const T* data = dataReader();
        Vec<uchar,3> elt;
        int index;
        for (int k = 0; k < nFrames; ++k) 
            for (int i = 0; i < height; ++i) {
                index = i + height*(x+width*k);
                for(int l = 0; l<3 ; l++) {
                    elt[l] = data[index];
                }
                out.at<Vec<uchar,3> >(i,k) = elt;
            }
        imwrite(path.string(),out);
    } else {
        fprintf(stderr,"Video must have 1 or 3 channels.\n");
        return;
    }
}

/** 
 * Export a frame to .jpg. For non-uint8 videos
 * @param[in] frame frame number.
 * @param[in] path output path.
 */
template <class T>
void Video<T>::exportScaledFrame(int frame,fs::path path, T min, T max) const {
    T a = 255/(max-min);
    T b = -min*a;

    if(nChannels == 3){
        Mat out(height,width,CV_8UC3);
        const T* data = dataReader();
        Vec<uchar,3> elt;
        int index;
        for (int j = 0; j < width; ++j)
            for (int i = 0; i < height; ++i) {
                index = i + height*(j+width*frame);
                for(int l = 0; l<nChannels ; l++) {
                    elt[l] = a*data[index+l*nVoxels]+b;
                }
                out.at<Vec<uchar,3> >(i,j) = elt;
            }
        imwrite(path.string(),out);
    } else if(nChannels == 1) {
        Mat out(height,width,CV_8UC3);
        const T* data = dataReader();
        Vec<uchar,3> elt;
        int index;
        for (int j = 0; j < width; ++j)
            for (int i = 0; i < height; ++i) {
                index = i + height*(j+width*frame);
                for(int l = 0; l<3 ; l++) {
                    elt[l] = a*data[index]+b;
                }
                out.at<Vec<uchar,3> >(i,j) = elt;
            }
        imwrite(path.string(),out);
    } else {
        fprintf(stderr,"Video must have 1 or 3 channels.\n");
        return;
    }
}

template <class T>
void Video<T>::exportVideo(fs::path path,string name) const{
    // Export frame images to a temporary directory
    fs::path tempDir       = path/(name+"-temp");
    fs::create_directories(tempDir);
    fs::path pathPrototype = (tempDir/(name+"-%04d.jpg"));
    boost::format format   = boost::format(pathPrototype.c_str());
    if(typeid(T) == typeid(unsigned char)) {
        for (int k = 0; k < nFrames; ++k) {
            exportFrame(k,(format % k).str());
        }
    } else {
        T min = this->min();
        T max = this->max();

        for (int k = 0; k < nFrames; ++k) {
            exportScaledFrame(k,(format % k).str(),min,max);
        }
    }

    // Clear previous file
    fs::path movieFile = path/(name+".mov");
    if(fs::exists(movieFile)){
        fs::remove(movieFile);
    }

    // Output final movie using ffmpeg
    boost::format outFormat = boost::format(" -filter:v scale=%d:%d") % (2*(width/2)) % (2*(height/2));
    string options          = "-vcodec libx264 -profile:v high ";
    options                += outFormat.str();
    string redirect         = ">/dev/null 2>&1";
    boost::format cmd       = boost::format("ffmpeg -i %s %s %s %s") % pathPrototype.c_str() % options % movieFile.c_str() % redirect ;
    system(cmd.str().c_str());

    // Cleanup
    fs::remove_all(tempDir);
}

template <class T>
void Video<T>::exportXTvideo(fs::path path,string name) const{
    // Export frame images to a temporary directory
    fs::path tempDir       = path/(name+"-temp");
    fs::create_directories(tempDir);
    fs::path pathPrototype = (tempDir/(name+"-%04d.jpg"));
    boost::format format   = boost::format(pathPrototype.c_str());
    if(typeid(T) == typeid(unsigned char)) {
        for (int y = 0; y < height; ++y) {
            exportXTslice(y,(format % y).str());
        }
    } else {
        IVideo copy(this->size());
        copy.copy(*this);
        for (int y = 0; y < height; ++y) {
            copy.exportXTslice(y,(format % y).str());
        }
    }

    // Clear previous file
    fs::path movieFile = path/(name+".mov");
    if(fs::exists(movieFile)){
        fs::remove(movieFile);
    }

    // Output final movie using ffmpeg
    boost::format outFormat = boost::format(" -filter:v scale=%d:%d") % (2*(width/2)) % (2*(nFrames/2));
    string options          = "-vcodec libx264 -profile:v high ";
    options                += outFormat.str();
    string redirect         = ">/dev/null 2>&1";
    boost::format cmd       = boost::format("ffmpeg -i %s %s %s %s") % pathPrototype.c_str() % options % movieFile.c_str() % redirect ;
    system(cmd.str().c_str());

    // Cleanup
    fs::remove_all(tempDir);
}

template <class T>
void Video<T>::exportYTvideo(fs::path path,string name) const{
    // Export frame images to a temporary directory
    fs::path tempDir       = path/(name+"-temp");
    fs::create_directories(tempDir);
    fs::path pathPrototype = (tempDir/(name+"-%04d.jpg"));
    boost::format format   = boost::format(pathPrototype.c_str());
    if(typeid(T) == typeid(unsigned char)) {
        for (int x = 0; x < width; ++x) {
            exportYTslice(x,(format % x).str());
        }
    } else {
        IVideo copy(this->size());
        copy.copy(*this);
        for (int x = 0; x < width; ++x) {
            copy.exportYTslice(x,(format % x).str());
        }
    }

    // Clear previous file
    fs::path movieFile = path/(name+".mov");
    if(fs::exists(movieFile)){
        fs::remove(movieFile);
    }

    // Output final movie using ffmpeg
    boost::format outFormat = boost::format(" -filter:v scale=%d:%d") % (2*(nFrames/2)) % (2*(height/2));
    string options          = "-vcodec libx264 -profile:v high ";
    options                += outFormat.str();
    string redirect         = ">/dev/null 2>&1";
    boost::format cmd       = boost::format("ffmpeg -i %s %s %s %s") % pathPrototype.c_str() % options % movieFile.c_str() % redirect ;
    system(cmd.str().c_str());

    // Cleanup
    fs::remove_all(tempDir);
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


#pragma mark - templated instatiations
template class Video<unsigned char>;
template class Video<int>;
template class Video<float>;
template class Video<double>;
