/* --------------------------------------------------------------------------
 * File:    WarpingField.cpp
 * Author:  Michael Gharbi <gharbi@mit.edu>
 * Created: 2014-01-29
 * --------------------------------------------------------------------------
 * 
 * Sub-class of Video implementing the specificities of the ST warping field
 * 
 * ------------------------------------------------------------------------*/

<<<<<<< HEAD
#include <cstdlib>
#include <cmath>
=======

#include "mcp/WarpingField.hpp"

#include <stdlib.h>
#include <math.h>
>>>>>>> 08e370661f7c4bf3290c9d5ccd5d5a000f9f11fc
#include <vector>
#include <fstream>
#include <typeinfo>

#include "mcp/WarpingField.hpp"


using namespace std;
typedef unsigned char uchar;


template <class T>
WarpingField<T>::WarpingField(fs::path path) {
    load(path);
}

template <class T>
WarpingField<T>::~WarpingField() {
    if( this->pData != NULL ) {
        delete[] this->pData;
        this->pData = NULL;
    }
}


/**
 * Load a .stw warping field binary file.
 * @param[in] path input file path
 */
template <class T>
void WarpingField<T>::load(fs::path path) {
    ifstream file;
    file.open(path.c_str(),ios::binary | ios::in);

    int type,h,w,nF,nC;
    file.read((char*)&type,sizeof(int));

    bool isValid = (type == 1 && typeid(T)==typeid(float));
    isValid = isValid || (type == 2 && typeid(T)==typeid(double));
    if(isValid){
        file.read((char*)&(h),sizeof(int));
        file.read((char*)&(w),sizeof(int));
        file.read((char*)&(nF),sizeof(int));
        file.read((char*)&(nC),sizeof(int));
        this->allocate(h,w,nF,nC);
        file.read((char*)this->pData,sizeof(T)*this->nElements);
    }else {
        fprintf(stderr,"Loading not implemented for this type of data, or incompatible data type.\n");
    }

    file.close();
}

/**
 * Save a .stw warping field binary file.
 * @param[in] path output file path
 */
template <class T>
void WarpingField<T>::save(fs::path path) {
    ofstream file;
    file.open(path.c_str(),ios::binary | ios::out);
    if(!file.is_open()) {
        printf(".stw saving failed to open the target file.\n");
    }
    int type = 0;
    if(typeid(T) == typeid(float)){
        type = 1;
    } else if(typeid(T) == typeid(double)){
        type = 2;
    } else{
        fprintf(stderr,"Saving not implemented for this type of data.\n");
    }
    file.write((char*)&type,sizeof(int));
    file.write((char*)&this->height,sizeof(int));
    file.write((char*)&this->width,sizeof(int));
    file.write((char*)&this->nFrames,sizeof(int));
    file.write((char*)&this->nChannels,sizeof(int));

    file.write((char*)this->pData,sizeof(T)*this->nElements);
    file.close();
}

template <class T>
void WarpingField<T>::setColors(int r, int g, int b, int k) {
    // NOTE: BGR format to agree with opencv code. Might change in the future.
    colorWheel[k][0] = b;
    colorWheel[k][1] = g;
    colorWheel[k][2] = r;
}

template <class T>
void WarpingField<T>::makeColorwheel() {
    // relative lengths of color transitions:
    // these are chosen based on perceptual similarity
    // (e.g.uone can distinguish more shades between red and yellow 
    //  than between yellow and green)
    int RY = 15;
    int YG = 6;
    int GC = 4;
    int CB = 11;
    int BM = 13;
    int MR = 6;
    const int MAXCOLS = RY + YG + GC + CB + BM + MR;
    colorWheel = vector<vector<int> >(MAXCOLS);
    for (size_t i = 0; i < colorWheel.size(); ++i) {
        colorWheel[i] = vector<int>(3);
    }
    int i;
    int k = 0;
    for (i = 0; i < RY; i++) setColors( 255, 255*i/RY,	0, k++);
    for (i = 0; i < YG; i++) setColors( 255-255*i/YG, 255, 0, k++);
    for (i = 0; i < GC; i++) setColors( 0, 255, 255*i/GC, k++);
    for (i = 0; i < CB; i++) setColors( 0, 255-255*i/CB, 255, k++);
    for (i = 0; i < BM; i++) setColors( 255*i/BM, 0, 255, k++);
    for (i = 0; i < MR; i++) setColors( 255, 0, 255-255*i/MR, k++);

}

template <class T>
void WarpingField<T>::computeColor(T u, T v, unsigned char *pMap) {
    int ncols      = colorWheel.size();
    float rad      = sqrt(u * u + v * v);
    float a        = atan2(-v, -u) / M_PI;
    float fk       = (a + 1.0) / 2.0 * (ncols-1);
    int k0         = (int)fk;
    int k1         = (k0 + 1) % ncols;
    float f        = fk - k0;

    for (int b = 0; b < 3; b++) {
        float col0 = colorWheel[k0][b] / 255.0;
        float col1 = colorWheel[k1][b] / 255.0;
        float col  = (1 - f) * col0 + f * col1;
        if (rad <= 1){
            col = 1 - rad * (1 - col); // increase saturation with radius
        } else {
            col *= .75; // out of range
        }
        pMap[b*this->nVoxels] = (int)(255.0 * col);
    }
}

/**
 * Save a color coded visualization of the warping field
 * @param[in] path output file path
 */
template <class T>
void WarpingField<T>::exportSpacetimeMap(fs::path path, string name, int maxAmplitude) {

    if( colorWheel.size() == 0) {
        makeColorwheel();
    }
    VideoSize s = this->size();
    s.nChannels = 3;
    IVideo spaceMap(s);
    IVideo timeMap(s);

    // determine motion range:
    T max_u = this->at(0,0,0,0);
    T max_v = this->at(0,0,0,1);
    T min_u = this->at(0,0,0,0);
    T min_v = this->at(0,0,0,1);

    T max_w = 0;
    T min_w = 0;
    if( this->nChannels >2) {
        max_w = this->at(0,0,0,2);
        min_w= this->at(0,0,0,2);
    }
    const T* pU = this->channelReader(0);
    const T* pV = this->channelReader(1);
    const T* pW;
    if( this->nChannels>2 ) {
        pW = this->channelReader(2);
    }else {
        pW = this->channelReader(1);
    }
    T maxrad = -1;
    for (int i = 0; i < this->nVoxels; ++i) {
        max_u = fmax(max_u,pU[i]);
        max_v = fmax(max_v,pV[i]);
        max_w = fmax(max_w,pW[i]);
        min_u = fmin(min_u,pU[i]);
        min_v = fmin(min_v,pV[i]);
        min_w = fmin(min_w,pW[i]);
        T rad = sqrt(pU[i]*pU[i]+pV[i]*pV[i]);
        maxrad = max(maxrad, rad);
    }

    printf("u:[%f,%f]\n",min_u,max_u);
    printf("v:[%f,%f]\n",min_v,max_v);
    printf("w:[%f,%f]\n",min_w,max_w);

    T maxtime = fmax(fabs(min_w),fabs(max_w));
    if (maxAmplitude > 0){
        maxrad = maxAmplitude;
        maxtime = maxAmplitude;
    }

    if (maxrad <= 0) {
        maxrad = 1;
    }
    if (maxtime <= 0) {
        maxtime = 1;
    }

    unsigned char *pSpace = spaceMap.dataWriter();
    for (int i = 0; i < this->nVoxels; ++i) {
        computeColor(pU[i]/maxrad, pV[i]/maxrad, pSpace);
        pSpace++;
    }

    if( this->nChannels >2) {
        unsigned char *pTime = timeMap.dataWriter();
        for (int i = 0; i < this->nVoxels; ++i) {
            computeColor(pW[i]/maxtime, pW[i]/maxtime, pTime);
            pTime++;
        }
    }
    spaceMap.exportVideo(path, name+"-space");
    timeMap.exportVideo(path, name+"-time");
}

/** 
 * Clear all data from the video container.
 */
// template <class T>
// void WarpingField<T>::clear() {
//     Video<T>::clear();
//     if( timeMap != NULL ) {
//         delete timeMap;
//         timeMap = NULL;
//     }
//     if( spaceMap != NULL ) {
//         delete spaceMap;
//         spaceMap = NULL;
//     }
// }

#pragma mark - Template instantiations
template class WarpingField<float>;
template class WarpingField<double>;
