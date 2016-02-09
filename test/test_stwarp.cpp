#include "gtest/gtest.h"
#include "video/Video.hpp"

#include "mcp/STWarp.hpp" 

class STWarpTest : public testing::Test {
};

TEST_F(STWarpTest, constructor) {
    IVideo A(320,160,30,3);
    IVideo B(320,160,30,3);

    WarpingField<float> uvw;
    STWarp<float> warper = STWarp<float>();

    STWarpParams params;
    warper.setParams(params);

    warper.setVideos(A,B);
    uvw = warper.computeWarp();
}
