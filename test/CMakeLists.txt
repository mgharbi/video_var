find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS})
set(test_bin ${bin}/test)

set(all_tests
    video
    stwarp
    nnfield
    nn_reconstruct
)

foreach(t ${all_tests})
    add_executable(test_${t} test_${t}.cpp)
    GTEST_ADD_TESTS(test_${t} "" test_${t}.cpp)
    set_target_properties(test_${t} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${test_bin})
    target_link_libraries(test_${t} 
            ${GTEST_BOTH_LIBRARIES}
            video
            mcp
    )
endforeach()
