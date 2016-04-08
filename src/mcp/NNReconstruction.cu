#include <cuda.h>
#include "cuda_utils.h"

#include "mcp/NNReconstruction.hpp"


// #define CUDA_CHECK(condition) \
//   #<{(| Code block avoids redefinition of cudaError_t error |)}># \
//   do { \
//     cudaError_t error = condition; \
//     CHECK_EQ(error, cudaSuccess) << " " << cudaGetErrorString(error); \
//   } while (0)

__global__ void reconstruct_kernel(int nVoxels,
        const int *nnf, const float* weight, nnf_data_t* db,
        int h, int w, int nF, int nC, int knn,
        int psz_space, int psz_time,
        nnf_data_t* buffer, int *aggregation_count
) 
{
    // y,x,t,c

    // idx = y + x*h + h*w*t;
    CUDA_KERNEL_LOOP(voxel, nVoxels) {
        int vx_t = voxel/(h*w);
        int vx_x = (voxel % (h*w)) / h ;
        int vx_y = voxel % h;
        if( vx_t >= nF-psz_time+1 || 
            vx_x >= w-psz_space+1 ||
            vx_y >= h-psz_space+1
        ) { // Out of bound
            return;
        }
        for (int k = 0; k < knn; ++k) {
            float curr_weight = weight[voxel + k*nVoxels];
            int target_x = nnf[voxel + 0*nVoxels + 3*k*nVoxels];
            int target_y = nnf[voxel + 1*nVoxels + 3*k*nVoxels];
            int target_t = nnf[voxel + 2*nVoxels + 3*k*nVoxels];

            for (int t = 0; t < psz_time; ++t)
            for (int x = 0; x < psz_space; ++x)
            for (int y = 0; y < psz_space; ++y)
            {
                int target_voxel = (y+target_y) + h*(x+target_x) + h*w*(t+target_t);
                for (int c = 0; c < nC; ++c) {
                    buffer[voxel+y + x*h + t*h*w + c*h*w*nF] += 
                        curr_weight * db[target_voxel+c*h*w*nF];
                }
                aggregation_count[voxel+y + x*h + t*h*w] += 1;
            } // patch pixel loop
        } // knn loop
    } // buffer px cudaloop
}


__global__ void normalize_kernel(int nVoxels,
        nnf_data_t* buffer, int *aggregation_count,
        int h, int w, int nF, int nC, int knn
) 
{
    CUDA_KERNEL_LOOP(voxel, nVoxels) {
        for (int c = 0; c < nC; ++c) {
            buffer[voxel + c*h*w*nF] = 
                knn * buffer[voxel + c*h*w*nF] / ((float)aggregation_count[voxel]);
        }
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

    // Copy input buffers to GPU
    int* nnf_d           = nullptr;
    float* weights_d     = nullptr;
    nnf_data_t* db_d     = nullptr;

    int sz_nnf = h*w*nF*3*knn*sizeof(int);
    cudaMalloc((void**) &nnf_d, sz_nnf);
    cudaMemcpy(nnf_d, nnf_->dataReader(),sz_nnf, cudaMemcpyHostToDevice);

    int sz_weights = h*w*nF*knn*sizeof(float);
    cudaMalloc((void**) &weights_d, sz_weights);
    cudaMemcpy(weights_d, w_->dataReader(),sz_weights, cudaMemcpyHostToDevice);

    int sz_db = h*w*nF*3*sizeof(nnf_data_t);
    cudaMalloc((void**) &db_d, sz_db);
    cudaMemcpy(db_d, db_->dataReader(),sz_db, cudaMemcpyHostToDevice);

    // Allocate GPU output buffers
    nnf_data_t* buffer_d = nullptr;;
    int* aggregation_d   = nullptr;;

    int sz_buffer = h*w*nF*nC*sizeof(nnf_data_t);
    cudaMalloc((void**) &buffer_d, sz_buffer);
    cudaMemset(buffer_d, 0, sz_buffer);

    int sz_aggreg = h*w*nF*1*sizeof(int);
    cudaMalloc((void**) &aggregation_d, sz_aggreg);
    cudaMemset(aggregation_d, 0, sz_aggreg);

    // Aggregate
    reconstruct_kernel<<<MCP_GPU_GET_BLOCKS(nVoxels), MCP_GPU_NUM_THREADS>>>(
        nVoxels,
        nnf_d, weights_d, db_d,
        h, w, nF, nC, knn,
        psz_space, psz_time,
        buffer_d, aggregation_d
    );

    // // Normalize
    normalize_kernel<<<MCP_GPU_GET_BLOCKS(nVoxels), MCP_GPU_NUM_THREADS>>>(
        nVoxels,
        buffer_d, aggregation_d,
        h, w, nF, nC, knn
    );

    // Copy buffer back to host
    Video<nnf_data_t> out(db_->size());
    cudaMemcpy(out.dataWriter(), buffer_d, sz_buffer, cudaMemcpyDeviceToHost);

    // Cleanup
    cudaFree(nnf_d);
    cudaFree(weights_d);
    cudaFree(db_d);
    cudaFree(buffer_d);
    cudaFree(aggregation_d);

    // CUDA_CHECK(cudaPeekAtLastError());

    return out;
}
