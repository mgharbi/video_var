include_directories(${inc})

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

# cuda_compile(mcp_cuda
#     NNReconstruction.cu
# )

add_library(mcp SHARED 
        ${libmcp_src} 
        # ${mcp_cuda}
)
target_link_libraries( mcp 
    video
    # ${CUDA_LIBRARIES}
)
include_directories("${inc}")
