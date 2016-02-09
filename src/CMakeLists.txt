# # Sources
# set(main_src main.cpp)
#
# add_executable(main ${main_src})
# target_link_libraries(main 
#     ${Boost_SYSTEM_LIBRARY}
#     ${Boost_FILESYSTEM_LIBRARY}
#     ${Boost_DATE_TIME_LIBRARY}
#     video
#     mcp
# )
#
add_subdirectory("mcp")
add_subdirectory("video")
add_subdirectory("mexfiles")

<<<<<<< HEAD
=======
# include_directories("${src}/video")
# include_directories("${src}/mcp")
# link_directories(${lib})
>>>>>>> 08e370661f7c4bf3290c9d5ccd5d5a000f9f11fc
