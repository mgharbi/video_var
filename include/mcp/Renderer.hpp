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

<<<<<<< HEAD:include/mcp/Renderer.hpp
#include <boost/filesystem.hpp>

#include "video/Video.hpp"
#include "video/VideoProcessing.hpp"

#include "mcp/STWarpParams.hpp"
#include "mcp/WarpingField.hpp"
=======

#include "video/Video.hpp"
#include "video/VideoProcessing.hpp"
#include <boost/filesystem.hpp>
#include "mcp/WarpingField.hpp"
#include "mcp/STWarpParams.hpp"
>>>>>>> 08e370661f7c4bf3290c9d5ccd5d5a000f9f11fc:include/mcp/Renderer.hpp

namespace fs = boost::filesystem;

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
