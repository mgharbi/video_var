#include <cuda.h>

#include "cuda_utils.h"
#include "mcp/NNReconstruction.hpp"

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

__global__ void reconstruct_kernel(int nVoxels, const int *nnf, const float* weight,
        int h, int w, int nF, int nC, int knn,
        int psz_space, int psz_time,
        nnf_data_t* buffer) 
{
    // y,x,t,c

    // idx = y + x*h + h*w*t;
    CUDA_KERNEL_LOOP(index, nVoxels) {
        // int t = index / (h*w);
        // int x = (index % (h*w)) / h;
        // int y = index % h;

        for (int k = 0; k < knn; ++k) {
            float curr_weight = weight[index + k*nVoxels];
            int target_x = nnf[index + 0*nVoxels + 3*k*nVoxels];
            int target_y = nnf[index + 1*nVoxels + 3*k*nVoxels];
            int target_t = nnf[index + 2*nVoxels + 3*k*nVoxels];

            for (int t = 0; t < psz_time; ++t)
            for (int x = 0; x < psz_space; ++x)
            for (int y = 0; y < psz_space; ++y)
            {
                for (int c = 0; c < nC; ++c) {
                    // buffer.at(voxel+y,x,t,c) +=  weight*((float)db_->at(target_y+y,target_x+x,target_t+t,c));
                    buffer[voxel+y + x*h + t*h*w + c*h*w*nF] += 
                        curr_weight *
                }
                aggregation_count.at(voxel+y,x,t,0) += 1;
            } // patch pixel loop
        } // knn loop
    }
}

Video<nnf_data_t> NNReconstruction::reconstruct_gpu() {
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
    nnf_data_t* buffer_d  = nullptr;

    int sz_nnf = h*w*nF*3*knn*sizeof(int);
    cudaMalloc((void**) &nnf_d, sz_nnf);
    cudaMemcpy(nnf_d, nnf_->dataReader(),sz_nnf, cudaMemcpyHostToDevice);

    int sz_weights = h*w*nF*knn*sizeof(float);
    cudaMalloc((void**) &weights_d, sz_weights);
    cudaMemcpy(weights_d, w_->dataReader(),sz_weights, cudaMemcpyHostToDevice);

    cudaMalloc((void**) &buffer_d, h*w*nF*nC*sizeof(nnf_data_t));

    // Aggregate
    reconstruct_kernel<<<MCP_GPU_GET_BLOCKS(nVoxels), MCP_GPU_NUM_THREADS>>>(
        nVoxels,
        nnf_d, weights_d,
        h, w, nF, nC, knn,
        psz_space, psz_time,
        buffer_d);

    // // Normalize

    // Copy buffer back to host
    Video<nnf_data_t> out(db_->size());

    // Cleanup
    cudaFree(nnf_d);
    cudaFree(weights_d);
    cudaFree(buffer_d);

    // CUDA_CHECK(cudaPeekAtLastError());

    return out;
}
