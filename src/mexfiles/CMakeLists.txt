# set
get_property(inc_dirs DIRECTORY PROPERTY INCLUDE_DIRECTORIES)

add_custom_target(
    stwarp_mex  ALL
    COMMAND mex -g stwarp.cpp -I${inc} -I/usr/local/include -L${lib} -lmcp -lvideo -outdir ${lib}/mex
    WORKING_DIRECTORY ${src}/mexfiles
    DEPENDS mcp
)
