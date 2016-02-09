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
#include <boost/filesystem.hpp>
// #include <boost/program_options.hpp>


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
    WarpingField(fs::path path);

    virtual ~WarpingField();

    // virtual void clear();

    // template <class T2> void copy(const WarpingField<T2>& source);

    // void exportSpacetimeMap(fs::path path, string name, int maxAmplitude=-1);

    // void save(fs::path path);
    // void load(fs::path path);

protected:
    // vector<vector<int> > colorWheel;

    // void makeColorwheel();
    // void setColors(int r, int g, int b, int k);
    // void computeColor(T u, T v, uchar *pix);
};

#pragma mark - type aliases
typedef WarpingField<float> FWarpingField;
typedef WarpingField<double> DWarpingField;

#endif /* end of include guard: WARPINGFIELD_HPP_EIWRCZTU */
