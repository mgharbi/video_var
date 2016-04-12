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

        // cout << nnf.at(i,j,k,0) << ", "
        //      << nnf.at(i,j,k,1) << ", "
        //      << nnf.at(i,j,k,2)  
        //      << " | " 
        //      << j << ", "
        //      << i << ", "
        //      << k  
        //      << " | " <<  cost.at(i,j,k) << endl;
             
        ASSERT_EQ(nnf.at(i,j,k,0), j);
        ASSERT_EQ(nnf.at(i,j,k,1), i);
        ASSERT_EQ(nnf.at(i,j,k,2), k);
    }
}

TEST_F(NNFieldTest, large_nnf) {
    int h = 50;
    int w = 50;
    int nF = 50;

    NNFieldParams params;
    params.knn = 16;
    params.patch_size_time = 11;
    params.patch_size_space = 11;
    params.verbosity = 0;
    params.propagation_iterations = 1;

    Video<nnf_data_t> A(h,w,nF,3);

    nnf_data_t * pData = A.dataWriter();
    for (int i = 0; i < A.elementCount(); ++i)
    {
        pData[i] = (rand() % 255)/255.0;
    }

    Video<nnf_data_t> B(A);


    NNField field(&A, &B, params);
    NNFieldOutput output = field.compute();
    Video<int32_t> &nnf  = output.nnf;

    // Check the first match is exact
    for (int i = 0; i < h-params.patch_size_space; ++i)
    for (int j = 0; j < w-params.patch_size_space; ++j)
    for (int k = 0; k < nF-params.patch_size_time; ++k)
    {

        ASSERT_EQ(nnf.at(i,j,k,0), j);
        ASSERT_EQ(nnf.at(i,j,k,1), i);
        ASSERT_EQ(nnf.at(i,j,k,2), k);
    }
}

TEST_F(NNFieldTest, large_nnf_gpu) {
    int h = 50;
    int w = 50;
    int nF = 50;

    NNFieldParams params;
    params.knn = 16;
    params.patch_size_time = 11;
    params.patch_size_space = 11;
    params.verbosity = 2;
    params.propagation_iterations = 1;

    Video<nnf_data_t> A(h,w,nF,3);

    nnf_data_t * pData = A.dataWriter();
    for (int i = 0; i < A.elementCount(); ++i)
    {
        pData[i] = (rand() % 255)/255.0;
    }

    Video<nnf_data_t> B(A);


    NNField field(&A, &B, params);
    NNFieldOutput output = field.compute_gpu();
    Video<int32_t> &nnf  = output.nnf;
    Video<nnf_data_t> &cost  = output.error;

    // Check the first match is exact
    for (int i = 0; i < h-params.patch_size_space; ++i)
    for (int j = 0; j < w-params.patch_size_space; ++j)
    for (int k = 0; k < nF-params.patch_size_time; ++k)
    {

        // cout << nnf.at(i,j,k,0) << ", "
        //      << nnf.at(i,j,k,1) << ", "
        //      << nnf.at(i,j,k,2)  
        //      << " | " 
        //      << j << ", "
        //      << i << ", "
        //      << k  
        //      << " | " <<  cost.at(i,j,k) << endl;
        // ASSERT_EQ(nnf.at(i,j,k,0), j);
        // ASSERT_EQ(nnf.at(i,j,k,1), i);
        // ASSERT_EQ(nnf.at(i,j,k,2), k);
    }
}
