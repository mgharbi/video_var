#include <iostream>

#include "gtest/gtest.h"
#include "video/Video.hpp"

#include "mcp/STWarp.hpp" 

using namespace std;

class STWarpTest : public testing::Test {
};

TEST_F(STWarpTest, zero_warp_field) {
    IVideo A(10,20,30,3);
    IVideo B(10,20,30,3);

    WarpingField<float> uvw;
    STWarp<float> warper = STWarp<float>();

    STWarpParams params;
    warper.setParams(params);

    warper.setVideos(A,B);
    uvw = warper.computeWarp();

    ASSERT_EQ(uvw.getHeight(), 10);
    ASSERT_EQ(uvw.getWidth(), 20);
    ASSERT_EQ(uvw.frameCount(), 30);
    ASSERT_EQ(uvw.channelCount(), 3);

    for (int i = 0; i < 10; ++i)
    for (int j = 0; i < 20; ++i)
    for (int k = 0; k < 30; ++k)
    {

        EXPECT_FLOAT_EQ(uvw.at(i,j,k,0), 0.0f);
        EXPECT_FLOAT_EQ(uvw.at(i,j,k,1), 0.0f);
        EXPECT_FLOAT_EQ(uvw.at(i,j,k,2), 0.0f);
    }
}

TEST_F(STWarpTest, nonzero_warp_field) {
    IVideo A(10,20,30,3);
    IVideo B(10,20,30,3);

    A.setAt(1.0f, 5,10,15,0);
    A.setAt(1.0f, 5,10,15,1);
    A.setAt(1.0f, 5,10,15,2);

    B.setAt(1.0f, 6,10,15,0);
    B.setAt(1.0f, 6,10,15,1);
    B.setAt(1.0f, 6,10,15,2);

    WarpingField<float> uvw;
    STWarp<float> warper = STWarp<float>();

    STWarpParams params;
    warper.setParams(params);

    warper.setVideos(A,B);
    uvw = warper.computeWarp();

    ASSERT_EQ(uvw.getHeight(), 10);
    ASSERT_EQ(uvw.getWidth(), 20);
    ASSERT_EQ(uvw.frameCount(), 30);
    ASSERT_EQ(uvw.channelCount(), 3);

    cout << uvw.at(4,10,15,1) << endl;
    cout << uvw.at(5,10,15,1) << endl;
    cout << uvw.at(6,10,15,1) << endl;

    // for (int i = 0; i < 10; ++i)
    // for (int j = 0; i < 20; ++i)
    // for (int k = 0; k < 30; ++k)
    // {
    //
    //     EXPECT_FLOAT_EQ(uvw.at(i,j,k,0), 0.0f);
    //     EXPECT_FLOAT_EQ(uvw.at(i,j,k,1), 0.0f);
    //     EXPECT_FLOAT_EQ(uvw.at(i,j,k,2), 0.0f);
    // }
}
