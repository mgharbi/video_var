/* --------------------------------------------------------------------------
 * File:    STWarp.hpp
 * Author:  Michael Gharbi <gharbi@mit.edu>
 * Created: 2014-01-29
 * --------------------------------------------------------------------------
 * 
 * 
 * 
 * ------------------------------------------------------------------------*/


#ifndef STWARP_HPP_QFL971LW
#define STWARP_HPP_QFL971LW

#include <cmath>
#include <vector>
#include <list>
#include <typeinfo>
#include <map>

#include "video/Video.hpp"
#include "video/VideoProcessing.hpp"

#include "mcp/STWarpParams.hpp"
#include "mcp/WarpingField.hpp"
#include "mcp/utils.hpp"


typedef float stwarp_video_t;
typedef multimap<int, pair<int, int> > NeighborhoodType;

using namespace std;

/**
 * Implements the space-time correspondence algorithm.
 */
template <class T>
class STWarp
{
public:
    STWarp();
    virtual ~STWarp();

    STWarpParams getParams();
    void setDefaultParams();
    void setParams(STWarpParams params);

    void setVideos(const Video<stwarp_video_t> &A, const Video<stwarp_video_t> &B);

    void setInitialWarpField(WarpingField<T> initial);

    WarpingField<T> computeWarp();

    vector<vector<int> > getPyramidSizes() const;
    void buildPyramid(vector<vector<int> > pyrSizes,
                      vector<Video<stwarp_video_t>*> &pyramidA,
                      vector<Video<stwarp_video_t>*> &pyramidB
                      ) const;
    void resampleWarpingField( WarpingField<T> &warpField, vector<int> dims ) const;
    void computePartialDerivatives( const WarpingField<T> &warpField, 
                                    Video<T> &Bx, 
                                    Video<T> &By, 
                                    Video<T> &Bt, 
                                    Video<T> &C);
    void multiscaleIteration( WarpingField<T>& warpField );
    void warpingIteration(  const WarpingField<T> &warpingField,
                            const Video<T> &Bx,
                            const Video<T> &By, 
                            const Video<T> &Bt, 
                            const Video<T> &C,
                            WarpingField<T> &dWarpField);
    void weightedLaplacian( const Video<T>& input, 
                            const Video<T>& weight, 
                            Video<T>& output);
    void denoiseWarpingField( WarpingField<T> &warpField);

    NeighborhoodType neighborhood;

    Video<T> occlusion;

private:
    Video<stwarp_video_t> *videoA;
    Video<stwarp_video_t> *videoB;
    WarpingField<T> *initialWarpField;

    vector<int> dimensions;
    STWarpParams params;


    void init();
    void initializeWarpField(const vector<int> &dimensions, WarpingField<T> &warpField);
    void computeSmoothCost(const Video<T> &warpDX,
                           const Video<T> &warpDY,
                           const Video<T> &warpDT,
                           const Video<T> &warpField,
                           Video<T> &smoothCost,
                           Video<T> &lapl);
    void computeOcclusion( const Video<T> &warpField,
                           const Video<T> &C,
                           Video<T> &occ);
    void computeDataCost(     const Video<T> &Bx,
                              const Video<T> &By,
                              const Video<T> &Bt,
                              const Video<T> &C,
                              const Video<T> &dWarpField,
                              const Video<T> &occlusion,
                              Video<T> &dataCost);
    void prepareLinearSystem( const Video<T> &Bx,
                              const Video<T> &By,
                              const Video<T> &Bt,
                              const Video<T> &C,
                              const Video<T> &lapl,
                              const Video<T> &dataCost,
                              Video<T> &CBx,
                              Video<T> &CBy,
                              Video<T> &CBt,
                              Video<T> &Bx2,
                              Video<T> &By2,
                              Video<T> &Bt2,
                              Video<T> &Bxy,
                              Video<T> &Bxt,
                              Video<T> &Byt
                             );
    void sor( const Video<T> &dataCost,
              const Video<T> &smoothCost,
              const Video<T> &lapl,
              const Video<T> &CBx,
              const Video<T> &CBy,
              const Video<T> &CBt,
              const Video<T> &Bx2,
              const Video<T> &By2,
              const Video<T> &Bt2,
              const Video<T> &Bxy,
              const Video<T> &Bxt,
              const Video<T> &Byt,
              Video<T> &dWarpField
             );
};

// Typedefs
typedef STWarp<float> FSTWarp;
typedef STWarp<double> DSTWarp;

#endif /* end of include guard: STWARP_HPP_QFL971LW */

