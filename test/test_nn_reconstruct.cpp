#include <iostream>

#include "gtest/gtest.h"
#include "video/Video.hpp"

#include "mcp/NNReconstruction.hpp" 

using namespace std;

class NNReconstructTest : public testing::Test {
};

TEST_F(NNReconstructTest, simple_nnf) {
    int h  = 5;
    int w  = 3;
    int nF = 3;
    int k  = 2;
    IVideo db(h,w,nF,3);
    Video<int> nnf(h,w,nF,3*k);
    Video<float> weight(h,w,nF,k);
    for (int i = 0; i < weight.elementCount(); ++i) {
        weight.at(i) = 1;
    }

    NNReconstructionParams params;
    params.patch_size_time = 1;
    params.patch_size_space = 1;
    unsigned char * pData = db.dataWriter();
    for (int i = 0; i < db.elementCount(); ++i)
    {
        pData[i] = rand() % 255;
    }

    NNReconstruction recons(&db,&nnf,&weight,params);
    IVideo out = recons.reconstruct();

    const unsigned char * pOut = out.dataReader();
    // nVoxels = out.voxelCount();
    for (int pt = 0; pt < nF-params.patch_size_time+1; ++pt)
    for (int px = 0; px < w-params.patch_size_space+1; ++px)
    for (int py = 0; py < h-params.patch_size_space+1; ++py)
    {
        int i = py + h*(px + w*pt);
        ASSERT_EQ(pOut[i], pData[0]);
    }
}
