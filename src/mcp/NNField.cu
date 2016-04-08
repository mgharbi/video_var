#include <cuda.h>
#include "cuda_utils.h"

#include "mcp/NNField.hpp"
#include <algorithm>
#include <cassert>

__global__ void initialize_kernel(int nVoxels,
        const nnf_data_t* video,
        const nnf_data_t* db,
        int h, int w, int nF, 
        int h_db, int w_db, int nF_db, 
        int nC, int knn,
        int psz_space, int psz_time,
        int* nnf, nnf_data_t *cost
) 
{
}

NNFieldOutput NNField::compute_gpu() {
    int knn = params_.knn;
    int psz_space = params_.patch_size_space;
    int psz_time  = params_.patch_size_time;

    int h       = video_->getHeight();  int h_valid  =  h - psz_space+1;
    int w       = video_->getWidth();   int w_valid  =  w - psz_space+1;
    int nF      = video_->frameCount(); int nF_valid = nF - psz_time+1;
    int nC      = video_->channelCount();
    int nVoxels = video_->voxelCount();

    // So that patches don't read out of database
    int h_db  = database_->getHeight(); int h_db_valid   =  h_db  - psz_space + 1;
    int w_db  = database_->getWidth(); int w_db_valid    =  w_db  - psz_space + 1;
    int nF_db = database_->frameCount(); int nF_db_valid = nF_db  - psz_time + 1;

    // Copy input buffers to GPU
    nnf_data_t* video_d = nullptr;
    nnf_data_t* db_d    = nullptr;;

    int sz_video = h*w*nF*3*sizeof(nnf_data_t);
    cudaMalloc((void**) &video_d, sz_video);
    cudaMemcpy(video_d, video_->dataReader(),sz_video, cudaMemcpyHostToDevice);

    int sz_db = h_db*w_db*nF_db*3*sizeof(nnf_data_t);
    cudaMalloc((void**) &db_d, sz_db);
    cudaMemcpy(db_d, database_->dataReader(),sz_db, cudaMemcpyHostToDevice);

    // Allocate GPU output buffers
    int* nnf_d          = nullptr;
    nnf_data_t* cost_d  = nullptr;

    int sz_nnf = h*w*nF*3*knn*sizeof(int);
    cudaMalloc((void**) &nnf_d, sz_nnf);
    cudaMemset(nnf_d, 0, sz_nnf);

    int sz_cost = h*w*nF*1*knn*sizeof(nnf_data_t);
    cudaMalloc((void**) &cost_d, sz_cost);
    cudaMemset(cost_d, 0, sz_cost);

    // init
    if(params_.verbosity > 0) {
        printf("+ NNF (gpu) initialization with size %dx%dx%d, ",h,w,nF);
        printf("patch size %dx%d\n",psz_space,psz_time);
    };

    initialize_kernel<<<MCP_GPU_GET_BLOCKS(nVoxels), MCP_GPU_NUM_THREADS>>>(
            nVoxels,
            video_d,
            db_d,
            h, w, nF, 
            h_db, w_db, nF_db, 
            nC, knn,
            psz_space, psz_time,
            nnf_d, cost_d
    );

    // Prepare host output buffers
    NNFieldOutput output(h,w,nF, knn);
    Video<int> &nnf          = output.nnf;
    Video<nnf_data_t> &error = output.error;

    // Copy buffers back to host
    cudaMemcpy(nnf.dataWriter(), nnf_d, sz_nnf, cudaMemcpyDeviceToHost);
    cudaMemcpy(error.dataWriter(), cost_d, sz_cost, cudaMemcpyDeviceToHost);

    // Cleanup
    cudaFree(nnf_d);
    cudaFree(cost_d);
    cudaFree(video_d);
    cudaFree(db_d);

    return output;
}
