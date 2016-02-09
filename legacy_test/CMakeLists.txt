add_definitions (-DBOOST_TEST_DYN_LINK)

add_executable(testVideo VideoTests.cpp)
target_link_libraries(testVideo 
    ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY}
    video
    boost_system-mt
)

add_executable(testVideoProcessing VideoProcessingTests.cpp)
target_link_libraries(testVideoProcessing 
    ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY}
    video
    boost_system-mt
)

add_executable(testSTWarp STWarpTests.cpp)
target_link_libraries(testSTWarp 
    ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY}
    mcp
    video
    opticalFlow
    boost_system-mt
)

include_directories("${src}/video")
include_directories("${src}/mcp")
include_directories("${src}/optical_flow")
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
