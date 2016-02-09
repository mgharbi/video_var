/* --------------------------------------------------------------------------
 * File:    Renderer.hpp
 * Author:  Michael Gharbi <gharbi@mit.edu>
 * Created: 2014-01-29
 * --------------------------------------------------------------------------
 * 
 * 
 * 
 * ------------------------------------------------------------------------*/

#ifndef RENDERER_HPP_NDCYHBTA
#define RENDERER_HPP_NDCYHBTA

#include <algorithm>

#include "video/Video.hpp"
#include "video/VideoProcessing.hpp"

#include "mcp/STWarpParams.hpp"
#include "mcp/WarpingField.hpp"


template <class T>
class Renderer
{
public:
    Renderer(STWarpParams params): params(params){};
    // virtual ~Renderer();
    IVideo render(const IVideo& videoA, 
                         const IVideo& videoB,
                         const WarpingField<T> &warpField,
                         double* exageration,
                         WarpingField<T> *flow = NULL);

protected:
    STWarpParams params;
};

#endif /* end of include guard: RENDERER_HPP_NDCYHBTA */
