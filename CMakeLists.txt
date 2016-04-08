cmake_minimum_required(VERSION 2.8)
project(MotionComparison)

# No optimization, gdb-friendly, warnings
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -fPIC -pthread")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0 -Wall -ggdb3")
# set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_MACOSX_RPATH 0)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules")

find_package(OpenMP QUIET)
find_package(CUDA QUIET REQUIRED)

if(CUDA_FOUND)
    list(APPEND CUDA_NVCC_FLAGS "--compiler-options;-O3;-DVERBOSE;-std=c++11")
endif(CUDA_FOUND)

if(OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

message("CMAKE_BUILD_TYPE is ${CMAKE_BUILD_TYPE}")
message("Compiler flags: ${CMAKE_CXX_FLAGS}")
if(${CMAKE_BUILD_TYPE} STREQUAL RELEASE)
    message("Compiler flags (release): ${CMAKE_CXX_FLAGS_RELEASE}")
else()
    message("Compiler flags (debug): ${CMAKE_CXX_FLAGS_DEBUG}")
endif()


# Directories
set(src "${CMAKE_CURRENT_SOURCE_DIR}/src")
set(test_dir "${CMAKE_CURRENT_SOURCE_DIR}/test")
set(bin "${CMAKE_CURRENT_SOURCE_DIR}/bin")
set(lib "${CMAKE_CURRENT_SOURCE_DIR}/lib")
set(inc "${CMAKE_CURRENT_SOURCE_DIR}/include")
set(matlab "${CMAKE_CURRENT_SOURCE_DIR}/matlab")
set(third_party "${CMAKE_CURRENT_SOURCE_DIR}/third_party")

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${lib})
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${lib})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${bin})

add_subdirectory(${src})

include_directories(${inc})

################################################################################
#                                   Testing                                    #
################################################################################

set(GTEST_ROOT ${third_party}/googletest/googletest)
enable_testing()
add_subdirectory(${test_dir})
# add_test(NAME testVideo COMMAND testVideo)
# add_test(NAME testVideoProcessing COMMAND testVideoProcessing)
# add_test(NAME testSTWarp COMMAND testSTWarp)

################################################################################
#                                  Debugging                                   #
################################################################################

# add_custom_command(
#     OUTPUT "${bin}/.lldbinit"
#     COMMAND ln -s ${CMAKE_CURRENT_SOURCE_DIR}/lldbinit ${bin}/.lldbinit
# )
# add_custom_target(
#     debugFile 
#     ALL
#     DEPENDS "${bin}/.lldbinit"
# )
#
