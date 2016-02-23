#include <iostream>

#include "gtest/gtest.h"
#include "video/Video.hpp"

#include "mcp/NNField.hpp" 

using namespace std;

class NNFieldTest : public testing::Test {
};

TEST_F(NNFieldTest, zero_warp_field) {
    IVideo A(10,20,30,3);

    unsigned char * pData = A.dataWriter();
    for (int i = 0; i < A.elementCount(); ++i)
    {
        pData[i] = rand() % 255;
    }


    IVideo B(A);

    NNFieldParams params;
    params.knn = 2;
    params.propagation_iterations = 3;

    NNField field(&A, &B, params);
    Video<int> nnf = field.compute();

    // Check the first match is exact
    for (int i = 0; i < 10-params.patch_size_space; ++i)
    for (int j = 0; j < 20-params.patch_size_space; ++j)
    for (int k = 0; k < 30-params.patch_size_time; ++k)
    {

        ASSERT_EQ(nnf.at(i,j,k,0), j);
        ASSERT_EQ(nnf.at(i,j,k,1), i);
        ASSERT_EQ(nnf.at(i,j,k,2), k);

    }

}
