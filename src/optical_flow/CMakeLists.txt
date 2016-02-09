
set(liboflow_src 
    OpticalFlow.cpp
    OpticalFlow.cpp
    GaussianPyramid.cpp
)

add_library(opticalFlow SHARED ${liboflow_src})

target_link_libraries(opticalFlow 
    ${OpenCV_LIBS}
    ${Boost_SYSTEM_LIBRARY}
    ${Boost_FILESYSTEM_LIBRARY}
)


include_directories("${src}/video")
