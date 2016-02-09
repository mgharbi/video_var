set(libmcp_src 
    STWarp.cpp
    STWarpParams.cpp
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
    ${Boost_PROGRAM_OPTIONS_LIBRARY}
)
<<<<<<< HEAD
=======

include_directories("${inc}")
>>>>>>> 08e370661f7c4bf3290c9d5ccd5d5a000f9f11fc
