# set
get_property(inc_dirs DIRECTORY PROPERTY INCLUDE_DIRECTORIES)
message("ocv ${inc_dirs}")
message("L ${CMAKE_MODULE_LINKER_FLAGS}")

message("boost ${Boost_LIBRARIES}")

add_custom_target(
    stwarp_mex  ALL
    COMMAND mex stwarp.cpp -I${inc} -I/usr/local/include -L${lib} -lmcp -lvideo -lboost_system -lboost_filesystem -outdir ${lib}/mex
    WORKING_DIRECTORY ${src}/mexfiles
    DEPENDS mcp
)
