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

<<<<<<< HEAD:include/mcp/STWarp.hpp
=======
#include "mcp/STWarpParams.hpp"
#include "mcp/utils.hpp"
#include "video/Video.hpp"
#include "video/VideoProcessing.hpp"
#include "mcp/WarpingField.hpp"
>>>>>>> 08e370661f7c4bf3290c9d5ccd5d5a000f9f11fc:include/mcp/STWarp.hpp
#include <cmath>
#include <vector>
#include <list>
#include <typeinfo>

<<<<<<< HEAD:include/mcp/STWarp.hpp
#include <boost/filesystem.hpp>

#include "video/Video.hpp"
#include "video/VideoProcessing.hpp"

#include "mcp/STWarpParams.hpp"
#include "mcp/WarpingField.hpp"
#include "mcp/utils.hpp"

=======
>>>>>>> 08e370661f7c4bf3290c9d5ccd5d5a000f9f11fc:include/mcp/STWarp.hpp
// #include "Image.h"
// #include "OpticalFlow.h"


namespace fs = boost::filesystem;
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
    STWarp(fs::path outPath);
    virtual ~STWarp();

    STWarpParams getParams();
    void setDefaultParams();
    void setParams(STWarpParams params);
    void loadParams(fs::path path);

    void setVideos(IVideo *A, IVideo *B);
    // void loadMasks(fs::path pathA, fs::path pathB);

    void setInitialWarpField(WarpingField<T> initial);

    WarpingField<T> computeWarp();

    vector<vector<int> > getPyramidSizes() const;
    void buildPyramid(vector<vector<int> > pyrSizes,
                      vector<IVideo*> &pyramidA,
                      vector<IVideo*> &pyramidB,
                      vector<IVideo*> &pyrMaskA,
                      vector<WarpingField<T>*> &pyrFlowA
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
    void denoiseWarpingField( WarpingField<T> &warpField, const Video<T> &occlusion );

    // void computeOpticalFlow();
    void thresholdFlow(double thresh);

    WarpingField<T> *flowA;
    NeighborhoodType neighborhood;

    Video<T> occlusion;

private:
    IVideo *videoA;
    IVideo *videoB;
    IVideo *maskA;
    WarpingField<T> *initialWarpField;

    vector<int> dimensions;
    STWarpParams params;

    void init();
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

