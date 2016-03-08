#include "mcp/NNReconstruction.hpp"
#include <cuda.h>

const int MCP_GPU_NUM_THREADS = 512;
inline int MCP_GPU_GET_BLOCKS(const int N) {
  return (N + MCP_GPU_NUM_THREADS - 1) / MCP_GPU_NUM_THREADS;
}

// #define CUDA_CHECK(condition) \
//   #<{(| Code block avoids redefinition of cudaError_t error |)}># \
//   do { \
//     cudaError_t error = condition; \
//     CHECK_EQ(error, cudaSuccess) << " " << cudaGetErrorString(error); \
//   } while (0)

__global__ void reconstruct_kernel(const int *nnf, const float* weight,
        int h, int w, int nF, int nC, int knn,
        int psz_space, int psz_time,
        float* buffer) 
{
}

IVideo NNReconstruction::reconstruct_gpu() {
    int h  = db_->getHeight();
    int w  = db_->getWidth();
    int nF = db_->frameCount();
    int nC = db_->channelCount();

    int knn = params_.knn;

    int psz_space = params_.patch_size_space;
    int psz_time  = params_.patch_size_time;

    int nVoxels = h*w*nF;

    // Copy buffers to GPU
    int* nnf_d       = nullptr;
    float* weights_d = nullptr;
    float* buffer_d  = nullptr;

    int sz_nnf = h*w*nF*3*knn*sizeof(int);
    cudaMalloc((void**) &nnf_d, sz_nnf);
    cudaMemcpy(nnf_d, nnf_->dataReader(),sz_nnf, cudaMemcpyHostToDevice);

    int sz_weights = h*w*nF*knn*sizeof(int);
    cudaMalloc((void**) &weights_d, sz_weights);
    cudaMemcpy(weights_d, w_->dataReader(),sz_weights, cudaMemcpyHostToDevice);

    cudaMalloc((void**) &buffer_d, h*w*nF*nC);

    // Aggregate
    reconstruct_kernel<<<MCP_GPU_GET_BLOCKS(nVoxels), MCP_GPU_NUM_THREADS>>>(
        nnf_d, weights_d,
        h, w, nF, nC, knn,
        psz_space, psz_time,
        buffer_d);

    // Normalize

    // Copy buffer back to host
    IVideo out(db_->size());

    // Cleanup
    cudaFree(nnf_d);
    cudaFree(weights_d);
    cudaFree(buffer_d);

    // CUDA_CHECK(cudaPeekAtLastError());

    return out;
}
