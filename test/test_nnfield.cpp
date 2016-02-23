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

    for (int i = 0; i < 10-params.patch_size_space; ++i)
    for (int j = 0; j < 20-params.patch_size_space; ++j)
    for (int k = 0; k < 30-params.patch_size_time; ++k)
    {

        ASSERT_EQ(nnf.at(i,j,k,0), j);
        ASSERT_EQ(nnf.at(i,j,k,1), i);
        ASSERT_EQ(nnf.at(i,j,k,2), k);

        // bool is_different = false;
        // is_different = is_different || (nnf.at(i,j,k,3) != nnf.at(i,j,k,0));
        // is_different = is_different || (nnf.at(i,j,k,4) != nnf.at(i,j,k,1));
        // is_different = is_different || (nnf.at(i,j,k,5) != nnf.at(i,j,k,2));

        // if(!is_different) {
        //     cout << nnf.at(i,j,k,0) << endl;
        //     cout << nnf.at(i,j,k,1) << endl;
        //     cout << nnf.at(i,j,k,2) << endl;
        //     cout << nnf.at(i,j,k,3) << endl;
        //     cout << nnf.at(i,j,k,4) << endl;
        //     cout << nnf.at(i,j,k,5) << endl;
        // }
        //
        // ASSERT_TRUE(is_different);
    }

}
