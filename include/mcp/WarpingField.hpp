/* --------------------------------------------------------------------------
 * File:    WarpingField.hpp
 * Author:  Michael Gharbi <gharbi@mit.edu>
 * Created: 2014-01-29
 * --------------------------------------------------------------------------
 * 
 * Sub-class of Video implementing the specificities of the ST warping field
 * 
 * ------------------------------------------------------------------------*/


#ifndef WARPINGFIELD_HPP_EIWRCZTU
#define WARPINGFIELD_HPP_EIWRCZTU

#include "video/Video.hpp"


/*
 * Extension to the Video class handling the specificities of a space-time
 * warping field volume.
 */
template <typename T>
class WarpingField : public Video<T>
{
public:
    WarpingField() : Video<T>(){};
    WarpingField(int height, int width, int nFrames, int nChannels) : 
        Video<T>(height, width, nFrames, nChannels){};
    WarpingField(VideoSize s) : Video<T>(s) {};
    WarpingField(const WarpingField& source): Video<T>(source){};

    virtual ~WarpingField();


protected:
};

typedef WarpingField<float> FWarpingField;
typedef WarpingField<double> DWarpingField;

#endif /* end of include guard: WARPINGFIELD_HPP_EIWRCZTU */
