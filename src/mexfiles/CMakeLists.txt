# set
get_property(inc_dirs DIRECTORY PROPERTY INCLUDE_DIRECTORIES)

set(mexfiles 
    stwarp
    knnfield
    reconstruct_from_knnf
)

foreach( f ${mexfiles})
add_custom_target(
    ${f}_mex  ALL
    COMMAND mex -g ${f}.cpp -I${inc} -I/usr/local/include -L${lib} -lmcp -lvideo -outdir ${lib}/mex
    WORKING_DIRECTORY ${src}/mexfiles
    DEPENDS mcp
)
endforeach()
