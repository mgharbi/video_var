<<<<<<< HEAD
# set
get_property(inc_dirs DIRECTORY PROPERTY INCLUDE_DIRECTORIES)
message("ocv ${inc_dirs}")
message("L ${CMAKE_MODULE_LINKER_FLAGS}")

add_custom_target(
    stwarp_mex  ALL
    COMMAND mex stwarp.cpp -I${inc} -I/usr/local/include -L${lib} -lmcp -lvideo -outdir ${lib}/mex
    WORKING_DIRECTORY ${src}/mexfiles
    DEPENDS mcp
)

=======
set( mex_src
    stwarp.cpp
)

add_custom_target(
    stwarp_mex ALL
    COMMAND mex ${mex_src} -I${inc} -L${lib} -lmcp -outdir ${lib}/mex
    WORKING_DIRECTORY ${src}/mexfiles
    DEPENDS mcp
)
>>>>>>> 08e370661f7c4bf3290c9d5ccd5d5a000f9f11fc
