#include <iostream>

#include "gtest/gtest.h"
#include "video/Video.hpp"

#include "mcp/NNField.hpp" 

using namespace std;

class NNFieldTest : public testing::Test {
};

TEST_F(NNFieldTest, simple_nnf) {
    Video<nnf_data_t> A(10,20,30,3);

    nnf_data_t * pData = A.dataWriter();
    for (int i = 0; i < A.elementCount(); ++i)
    {
        pData[i] = (rand() % 255)/255.0;
    }


    Video<nnf_data_t> B(A);

    NNFieldParams params;
    params.knn = 2;
    params.verbosity = 0;
    params.propagation_iterations = 3;

    NNField field(&A, &B, params);
    NNFieldOutput output = field.compute();
    Video<int32_t> &nnf  = output.nnf;
    // Video<nnf_data_t> &cost  = output.error;

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

TEST_F(NNFieldTest, simple_nnf_gpu) {
    Video<nnf_data_t> A(10,20,30,3);

    nnf_data_t * pData = A.dataWriter();
    for (int i = 0; i < A.elementCount(); ++i)
    {
        pData[i] = (rand() % 255)/255.0;
    }


    Video<nnf_data_t> B(A);

    NNFieldParams params;
    params.knn = 2;
    // params.verbosity = 0;
    params.propagation_iterations = 3;

    NNField field(&A, &B, params);
    NNFieldOutput output = field.compute_gpu();
    Video<int32_t> &nnf  = output.nnf;
    Video<nnf_data_t> &cost  = output.error;

    // Check the first match is exact
    for (int i = 0; i < 10-params.patch_size_space; ++i)
    for (int j = 0; j < 20-params.patch_size_space; ++j)
    for (int k = 0; k < 30-params.patch_size_time; ++k)
    {

        // cout << nnf.at(i,j,k,0) << "; " << cost.at(i,j,k,0) << endl;
        // ASSERT_EQ(nnf.at(i,j,k,0), j);
        // ASSERT_EQ(nnf.at(i,j,k,1), i);
        // ASSERT_EQ(nnf.at(i,j,k,2), k);
    }
}
