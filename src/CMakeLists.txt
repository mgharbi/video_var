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

