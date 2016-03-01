set(libmcp_src 
    STWarp.cpp
    NNField.cpp
    NNReconstruction.cpp
    Renderer.cpp
    STWarp_warpingIterationAux.cpp
    STWarp_warpingIteration.cpp
    STWarp_denoising.cpp
    WarpingField.cpp
)

include_directories(${inc})
add_library(mcp SHARED ${libmcp_src})
target_link_libraries( mcp 
    video
)
include_directories("${inc}")
