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
    Video<nnf_data_t> db(h,w,nF,3);
    Video<int> nnf(h,w,nF,3*k);
    Video<float> weight(h,w,nF,k);
    for (int i = 0; i < weight.elementCount(); ++i) {
        weight.at(i) = 1;
    }

    NNReconstructionParams params;
    params.patch_size_time = 1;
    params.patch_size_space = 1;
    nnf_data_t * pData = db.dataWriter();
    for (int i = 0; i < db.elementCount(); ++i)
    {
        pData[i] = (rand() % 255)/255.0;
    }

    NNReconstruction recons(&db,&nnf,&weight,params);
    Video<nnf_data_t> out = recons.reconstruct();

    const nnf_data_t * pOut = out.dataReader();
    // nVoxels = out.voxelCount();
    for (int pt = 0; pt < nF-params.patch_size_time+1; ++pt)
    for (int px = 0; px < w-params.patch_size_space+1; ++px)
    for (int py = 0; py < h-params.patch_size_space+1; ++py)
    {
        int i = py + h*(px + w*pt);
        ASSERT_EQ(pOut[i], pData[0]);
    }
}

TEST_F(NNReconstructTest, cuda_nnf) {
    int h  = 5;
    int w  = 3;
    int nF = 3;
    int k  = 2;
    Video<nnf_data_t> db(h,w,nF,3);
    Video<int> nnf(h,w,nF,3*k);
    Video<float> weight(h,w,nF,k);
    for (int i = 0; i < weight.elementCount(); ++i) {
        weight.at(i) = 1;
    }

    NNReconstructionParams params;
    params.patch_size_time = 1;
    params.patch_size_space = 1;
    nnf_data_t * pData = db.dataWriter();
    for (int i = 0; i < db.elementCount(); ++i)
    {
        pData[i] = (rand() % 255)/255.0;
    }

    NNReconstruction recons(&db,&nnf,&weight,params);
    Video<nnf_data_t> out = recons.reconstruct_gpu();

    const nnf_data_t * pOut = out.dataReader();
    for (int pt = 0; pt < nF-params.patch_size_time+1; ++pt)
    for (int px = 0; px < w-params.patch_size_space+1; ++px)
    for (int py = 0; py < h-params.patch_size_space+1; ++py)
    {
        int i = py + h*(px + w*pt);
        ASSERT_EQ(pOut[i], pData[0]);
    }
}

TEST_F(NNReconstructTest, larger_identity_nnf) {
    int h  = 100;
    int w  = 100;
    int nF = 100;
    int k  = 20;

    NNReconstructionParams params;
    params.patch_size_time = 11;
    params.patch_size_space = 11;

    Video<nnf_data_t> db(h,w,nF,3);
    Video<int> nnf(h,w,nF,3*k);
    Video<float> weight(h,w,nF,k);
    for (int i = 0; i < weight.elementCount(); ++i) {
        weight.at(i) = 1;
    }

    // Set NNF
    for (int u = 0; u < k; ++u)
    for (int t = 0; t < nF-params.patch_size_time+1; ++t)
    for (int x = 0; x < w-params.patch_size_space+1; ++x)
    for (int y = 0; y < h-params.patch_size_space+1; ++y)
    {
        nnf.at(y,x,t,3*u)   = x;
        nnf.at(y,x,t,3*u+1) = y;
        nnf.at(y,x,t,3*u+2) = t;
    }

    nnf_data_t * pData = db.dataWriter();
    for (int i = 0; i < db.elementCount(); ++i)
    {
        pData[i] = (rand() % 255)/255.0;
    }

    NNReconstruction recons(&db,&nnf,&weight,params);
    Video<nnf_data_t> out = recons.reconstruct();

    const nnf_data_t * pOut = out.dataReader();
    for (int pt = 0; pt < nF-params.patch_size_time+1; ++pt)
    for (int px = 0; px < w-params.patch_size_space+1; ++px)
    for (int py = 0; py < h-params.patch_size_space+1; ++py)
    {
        int i = py + h*(px + w*pt);
        ASSERT_NEAR(pOut[i], pData[i], 1e-3);
    }
}

TEST_F(NNReconstructTest, larger_identity_nnf_gpu) {
    int h  = 100;
    int w  = 100;
    int nF = 100;
    int k  = 20;

    NNReconstructionParams params;
    params.patch_size_time = 11;
    params.patch_size_space = 11;

    Video<nnf_data_t> db(h,w,nF,3);
    Video<int> nnf(h,w,nF,3*k);
    Video<float> weight(h,w,nF,k);
    for (int i = 0; i < weight.elementCount(); ++i) {
        weight.at(i) = 1;
    }

    // Set NNF
    for (int u = 0; u < k; ++u)
    for (int t = 0; t < nF-params.patch_size_time+1; ++t)
    for (int x = 0; x < w-params.patch_size_space+1; ++x)
    for (int y = 0; y < h-params.patch_size_space+1; ++y)
    {
        nnf.at(y,x,t,3*u)   = x;
        nnf.at(y,x,t,3*u+1) = y;
        nnf.at(y,x,t,3*u+2) = t;
    }

    nnf_data_t * pData = db.dataWriter();
    for (int i = 0; i < db.elementCount(); ++i)
    {
        pData[i] = 2*(rand() % 127)/255.0;
    }

    NNReconstruction recons(&db,&nnf,&weight,params);
    Video<nnf_data_t> out = recons.reconstruct_gpu();

    const nnf_data_t * pOut = out.dataReader();
    for (int pt = 0; pt < nF-params.patch_size_time+1; ++pt)
    for (int px = 0; px < w-params.patch_size_space+1; ++px)
    for (int py = 0; py < h-params.patch_size_space+1; ++py)
    {
        int i = py + h*(px + w*pt);
        EXPECT_NEAR(pOut[i], pData[i], 1e-1);
        // cout  << "("
        //     << px << ","
        //     << py << ","
        //     << pt << ") "
        //     << pOut[i] << "; " << pData[i] << endl;
    }
}
