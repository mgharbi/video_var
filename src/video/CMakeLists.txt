set(libvideo_src 
    Video.cpp
    VideoProcessing.cpp
)

include_directories(${inc})
include_directories(${OpenCV_INCLUDE_DIRS})
add_library(video SHARED ${libvideo_src})

target_link_libraries(video 
    ${OpenCV_LIBS}
    ${Boost_LIBRARIES}
)

include_directories("${inc}")
