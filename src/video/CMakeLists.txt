set(libvideo_src 
    Video.cpp
)

include_directories(${inc})
include_directories(${OpenCV_INCLUDE_DIRS})
add_library(video SHARED ${libvideo_src})

include_directories("${inc}")
