#include <algorithm>
#include <cassert>
#include <cfloat>

#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>
// #include <math_functions.h>
#include <thrust/execution_policy.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>

#include "cuda/utils.h"
#include "cuda/error_check.h"
#include "mcp/NNField.hpp"

// ---- GPU datastructures ---------------------------------------

__device__ dim3 get_patch_coord(int linear, int d1, int d2) {
    dim3 out(0,0,0);
    out.z = linear/ (d1*d2);
    out.y = (linear% (d1*d2)) / d1;
    out.x = linear% d1;
    return out;
}

typedef struct MatchGPU {
    __host__ __device__ MatchGPU(nnf_data_t cost, int x, int y, int t) :
        cost(cost), x(x), y(y), t(t) {}
    __host__ __device__ MatchGPU() :
        cost(FLT_MAX), x(-1), y(-1), t(-1) {}
    nnf_data_t cost;
    int x;
    int y;
    int t;
} MatchGPU; // cost,x,y,t

typedef struct MatchGPUCompare {
    __host__ __device__ bool operator() (const MatchGPU& a, const MatchGPU& b)  const {
        return a.cost < b.cost;
    }
} MatchGPUCompare;

typedef struct MatchGPUEqualTo {
    __host__ __device__ bool operator() (const MatchGPU& a, const MatchGPU& b)  const {
        bool ret = a.x == b.x;
        ret &= a.y == b.y; 
        ret &= a.t == b.t; 
        return ret;
    }
} MatchGPUEqualTo;

// ---------------------------------------------------------------

__global__ void setup_rng_kernel(int nVoxels, curandState *state){
  CUDA_KERNEL_LOOP(voxel, nVoxels) {
      curand_init(1234, voxel, 0, &state[voxel]);
  }
}


__device__ nnf_data_t d_patch_cost(
        const nnf_data_t* video,
        const nnf_data_t* db,
        int h, int w, int nF, int nC,
        int vx_y,int vx_x,int vx_t,
        int  y_db,int x_db,int t_db,
        int psz_space, int psz_time
) {
    nnf_data_t cost = 0;
    for (int t = 0; t < psz_time; ++t) 
    for (int x = 0; x < psz_space; ++x) 
    for (int y = 0; y < psz_space; ++y) 
    {
        int vx_video = y+vx_y + h*( x+vx_x + w*(t+vx_t));
        int vx_db = y+y_db + h*( x+x_db + w*(t+t_db));
        for( int c = 0 ; c < nC ; c++ ){
            nnf_data_t d = video[vx_video+c*h*w*nF] - db[vx_db+c*h*w*nF];
            cost += d*d;
        }
    }
    return cost;
}



__device__ void d_improve_knn(
    const nnf_data_t *video,const nnf_data_t *db,
    int h, int w, int nF, int nC, int knn,
    int psz_space, int psz_time,
    int vx_y, int vx_x, int vx_t,
    int y_p, int x_p, int t_p,
    MatchGPU *current_best
) 
{
    nnf_data_t candidate_dist = d_patch_cost(
        video, db,
        h, w, nF, nC,
        vx_y,vx_x,vx_t,
        y_p,x_p,t_p,
        psz_space, psz_time
    );

    // worst match
    nnf_data_t err = current_best[knn-1].cost;

    // Are we better than the worst candidate so far?
    // if( candidate_dist < err) { // we have a better match
        MatchGPU mnew = MatchGPU(candidate_dist,x_p,y_p,t_p);

        // Insert new match, if not in the list already
        MatchGPUEqualTo eq;
        bool unseen = true;
        // for(int k = 0; k < knn; ++k) {
        //     bool equals = eq(mnew, current_best[k]);
        //     unseen = unseen && !equals;
        // }
        // if(!unseen) { return; #<{(| already in, do nothing |)}># }

        // Otherwise add it to the list of matches, and sort (only the knn first are valid)
        // if(vx_x == 0 && vx_y == 0 && vx_t == 0) {
        //     printf("before\n");
        //     for (int k = 0; k < knn+1; ++k)
        //     {
        //         MatchGPU m = current_best[k];
        //         printf("%d | %d %d %d | %f\n", k, m.x, m.y, m.t, m.cost);
        //     }
        // }

        // FIXME: THIS IS SLOW
        current_best[knn] = mnew;
        // thrust::sort(thrust::seq, current_best, current_best+knn+1, MatchGPUCompare());
        current_best[knn] = MatchGPU(); // invalidate the worst match

        // if(vx_x == 0 && vx_y == 0 && vx_t == 0) {
        //     printf("after\n");
        //     for (int k = 0; k < knn+1; ++k)
        //     {
        //         MatchGPU m = current_best[k];
        //         printf("%d | %d %d %d | %f\n", k, m.x, m.y, m.t, m.cost);
        //     }
        // }
    // }
}


__global__ void initialize_kernel(int nVoxels,
        const nnf_data_t* video,
        const nnf_data_t* db,
        int h, int w, int nF,
        int h_db_valid, int w_db_valid, int nF_db_valid, 
        int nC, int knn,
        int psz_space, int psz_time,
        curandState *rng_state,
        int* nnf,
        nnf_data_t* cost
) 
{
    CUDA_KERNEL_LOOP(voxel, nVoxels) {
        dim3 xyt = get_patch_coord(voxel, h, w);
        int vx_x = xyt.x;
        int vx_y = xyt.y;
        int vx_t = xyt.z;

        if( vx_t >= nF-psz_time+1 || 
            vx_x >= w-psz_space+1 ||
            vx_y >= h-psz_space+1
        ) { // Out of bound
            return;
        }

        // Copy PRNG state to local memory
        curandState state = rng_state[voxel];

        for (int k = 0; k < knn; ++k) {
            int t_db = curand_uniform(&state) * nF_db_valid;
            int x_db = curand_uniform(&state) * w_db_valid;
            int y_db = curand_uniform(&state) * h_db_valid;
            nnf[ voxel*3*knn + 3*k + 0 ] = x_db;
            nnf[ voxel*3*knn + 3*k + 1 ] = y_db;
            nnf[ voxel*3*knn + 3*k + 2 ] = t_db;
        } // knn loop

        // Copy back PRNG state to global memory
        rng_state[voxel] = state;
    }
}


__global__ void update_cost(int nVoxels,
        const nnf_data_t* video,
        const nnf_data_t* db,
        int h, int w, int nF,
        int h_db_valid, int w_db_valid, int nF_db_valid, 
        int nC, int knn,
        int psz_space, int psz_time,
        curandState *rng_state,
        int* nnf,
        nnf_data_t* cost
) 
{
    int vx_x = blockIdx.x;
    int vx_y = blockIdx.y;
    int vx_t = blockIdx.z;

    // if( vx_t >= nF-psz_time+1 || 
    //     vx_x >= w-psz_space+1 ||
    //     vx_y >= h-psz_space+1
    // ) { // Out of bound
    //     return;
    // }

    int tid      = threadIdx.x;
    int stride   = blockDim.x;
    int n_passes = ceil(((float) psz_space*psz_space*psz_time ) / stride);
    int psz      = psz_space*psz_space*psz_time;

    __shared__ float c;
    __shared__ nnf_data_t patch[TILE_W*TILE_H*3];
    __shared__ int neighbors[32*3];

    int voxel = vx_y + h*(vx_x + w*vx_t);

    // Fill in neighbors
    if (tid < knn) {
        neighbors[3*tid + 0] = nnf[voxel*3*knn + 3*tid + 0];
        neighbors[3*tid + 1] = nnf[voxel*3*knn + 3*tid + 1];
        neighbors[3*tid + 2] = nnf[voxel*3*knn + 3*tid + 2];
        cost[voxel*knn + tid] = 0;
    }

    // Fill in patch in shared memory
    for (int pass = 0; pass < n_passes; ++pass)
    {
        if(tid+pass*stride > psz) {
            continue;
        }
        dim3 xyt = get_patch_coord(tid + pass*stride,psz_space,psz_space);
        for( int chan = 0 ; chan < nC ; chan++ ){
            int vx_video = xyt.y+vx_y + h*( xyt.x+vx_x + w*(xyt.z+vx_t));
            patch[xyt.y+TILE_H*(xyt.x+TILE_W*chan)] = video[vx_video+chan*h*w*nF];
        }
    }

    for (int pass = 0; pass < n_passes; ++pass)
    {
        if(tid+pass*stride > psz) {
            continue;
        }
        dim3 xyt = get_patch_coord(tid + pass*stride, psz_space, psz_space);
        for (int k = 0; k < knn; ++k) {
             // zero out the aggregator
            if (tid == 0) {
                c = 0;
            }
            __syncthreads();
                int x_db = neighbors[3*k + 0];
                int y_db = neighbors[3*k + 1];
                int t_db = neighbors[3*k + 2];

                int vx_db = xyt.y+y_db + h*( xyt.x+x_db + w*(xyt.z+t_db));
                for( int chan = 0 ; chan < nC ; chan++ ){
                    nnf_data_t d = patch[xyt.y + TILE_H*(xyt.x  +TILE_W*chan)] 
                        - db[vx_db + chan*h*w*nF];
                    c+= d*d;
                }
            __syncthreads();
            cost[voxel*knn + k] += c;
        } // knn loop
    } // pass
}


__global__ void pm_rs_kernel(int nVoxels,
        const nnf_data_t* video,
        const nnf_data_t* db,
        int h, int w, int nF,
        int h_db_valid, int w_db_valid, int nF_db_valid, 
        int nC, int knn,
        int psz_space, int psz_time,
        curandState *rng_state,
        MatchGPU* bestmatches
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

        // Fetch best
        MatchGPU *current_best = bestmatches + (knn+1)*voxel; // 1 extra for insertion/deletion
        
        // Copy PRNG state to local memory
        curandState state = rng_state[voxel];

        int rs_start = max(w, h); 
        int rt_start = nF;

        int mag =  rs_start;
        int mag_time = rt_start;
        while (mag >= 1 || mag_time >= 1)
        {
            if(mag >= 1) {
                mag /= 2;
            }

            if(mag_time >= 1) {
                mag_time /= 2;
            }
            for (int k = 0; k < knn; ++k) {
                int x_best = current_best[k].x;
                int y_best = current_best[k].y;
                int t_best = current_best[k].t;

                /* Sampling window */
                int y_min = max(y_best-mag, 0);
                int x_min = max(x_best-mag, 0);
                int t_min = max(t_best-mag_time, 0); 
                int y_max = min(y_best+mag+1,h_db_valid);
                int x_max = min(x_best+mag+1,w_db_valid);
                int t_max = min(t_best+mag_time+1,nF_db_valid);

                // New random proposal from the region
                int y_p    = y_min + curand_uniform(&state) * (y_max-y_min);
                int x_p    = x_min + curand_uniform(&state) * (x_max-x_min);
                int t_p    = t_min + curand_uniform(&state) * (t_max-t_min);

                d_improve_knn(video, db, h, w, nF, nC, knn, psz_space, psz_time,
                    vx_y, vx_x, vx_t, y_p, x_p, t_p, current_best);
            } // knn
        }

        // Copy back PRNG state to global memory
        rng_state[voxel] = state;

    } // buffer px cudaloop
}


__global__ void pm_propag_kernel(int nVoxels,
        const nnf_data_t* video,
        const nnf_data_t* db,
        int h, int w, int nF,
        int h_db_valid, int w_db_valid, int nF_db_valid, 
        int nC, int knn,
        int psz_space, int psz_time,
        int *nnf_d, nnf_data_t *cost_d,
        int *nnf_tmp_d, nnf_data_t *cost_tmp_d,
        int jump
) 
{
    // NOTE: assumes the kernel is realized on the correct domain
    int vx_x = blockIdx.x;
    int vx_y = blockIdx.y;
    int vx_t = blockIdx.z;

    int tid      = threadIdx.x;
    int stride   = blockDim.x;
    int psz      = psz_space*psz_space*psz_time;
    int n_passes = ceil(((float) psz ) / stride);

    extern __shared__ int shared_ptr[];
    int knn_offset = psz_space*psz_space*psz_time*nC;
    int cost_offset = 3*knn;
    nnf_data_t *patch = (nnf_data_t*) shared_ptr;
    int *best_matches = (int*)(patch+knn_offset);
    nnf_data_t *best_costs = (nnf_data_t*)(best_matches+cost_offset);

    int voxel = vx_y + h*(vx_x + w*vx_t);

    // Fill in current best NN
    if(tid < knn) {
        best_matches[3*tid + 0] = 0;
        best_matches[3*tid + 1] = 0;
        best_matches[3*tid + 2] = 0;
        best_costs[tid] = 0;
    }

    // Fill in patch in shared memory
    for (int pass = 0; pass < n_passes; ++pass)
    {
        if(tid+pass*stride > psz) {
            continue;
        }
        dim3 xyt = get_patch_coord(tid + pass*stride,psz_space,psz_space);
        for( int chan = 0 ; chan < nC ; chan++ ){
            int vx_video = voxel + xyt.y + h*( xyt.x + w*xyt.z);
            patch[xyt.y + psz_space*(xyt.x+psz_space*chan)] = video[vx_video + chan*h*w*nF];
        }
    }
    __syncthreads();

    // Propagate x
    if(vx_x - jump >= 0) {
        int voxel_p = voxel - jump*h;
        for (int k = 0; k < knn; ++k) { // get k next neighbors
            int x_p = best_matches[voxel_p*knn + 3*k + 0] + jump;
            int y_p = best_matches[voxel_p*knn + 3*k + 1];
            int t_p = best_matches[voxel_p*knn + 3*k + 2];
            if(x_p < w_db_valid) {
                d_improve_knn(video, db, h, w, nF, nC, knn, psz_space, psz_time,
                    vx_y, vx_x, vx_t, y_p, x_p, t_p, current_best);
            }
        }
    }

    // for (int pass = 0; pass < n_passes; ++pass)
    // {
    //     if(tid+pass*stride > psz) {
    //         continue;
    //     }
    //     dim3 xyt = get_patch_coord(tid + pass*stride, psz_space, psz_space);
    //     for (int k = 0; k < knn; ++k) {
    //          // zero out the aggregator
    //         if (tid == 0) {
    //             c = 0;
    //         }
    //         __syncthreads();
    //             int x_db = neighbors[3*k + 0];
    //             int y_db = neighbors[3*k + 1];
    //             int t_db = neighbors[3*k + 2];
    //
    //             int vx_db = xyt.y+y_db + h*( xyt.x+x_db + w*(xyt.z+t_db));
    //             for( int chan = 0 ; chan < nC ; chan++ ){
    //                 nnf_data_t d = patch[xyt.y + TILE_H*(xyt.x  +TILE_W*chan)] 
    //                     - db[vx_db + chan*h*w*nF];
    //                 c+= d*d;
    //             }
    //         __syncthreads();
    //         cost[voxel*knn + k] += c;
    //     } // knn loop
    // } // pass
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
    int h_db  = database_->getHeight();  int h_db_valid  =  h_db  - psz_space + 1;
    int w_db  = database_->getWidth();   int w_db_valid  =  w_db  - psz_space + 1;
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

    int* nnf_d         = nullptr;
    nnf_data_t* cost_d = nullptr;
    int* nnf_tmp_d         = nullptr;
    nnf_data_t* cost_tmp_d = nullptr;
    int sz_nnf         = h*w*nF*3*knn*sizeof(int);
    int sz_cost        = h*w*nF*knn*sizeof(nnf_data_t);
    cudaMalloc((void**) &nnf_d, sz_nnf);
    cudaMalloc((void**) &cost_d, sz_cost);
    cudaMemset(nnf_d, 0, sz_cost);
    cudaMemset(cost_d, 0, sz_cost);
    cudaMalloc((void**) &nnf_tmp_d, sz_nnf);
    cudaMalloc((void**) &cost_tmp_d, sz_cost);
    cudaMemset(nnf_tmp_d, 0, sz_nnf);
    cudaMemset(cost_tmp_d, 0, sz_cost);

    // init
    if(params_.verbosity > 0) {
        printf("+ NNF (gpu) initialization with size %dx%dx%d, ",h,w,nF);
        printf("patch size %dx%d\n",psz_space,psz_time);
    };

    // Prepare random number generator
    curandState *d_state;
    cudaMalloc(&d_state, nVoxels*sizeof(curandState));
    setup_rng_kernel<<<GPU_GET_BLOCKS(nVoxels), GPU_THREADS>>>(nVoxels, d_state);

    initialize_kernel<<<GPU_GET_BLOCKS(nVoxels), GPU_THREADS>>>(
            nVoxels,
            video_d,
            db_d,
            h, w, nF,
            h_db_valid, w_db_valid, nF_db_valid, 
            nC, knn,
            psz_space, psz_time,
            d_state,
            nnf_d,
            cost_d
    );
    dim3 tpb_cost(GPU_THREADS, 1, 1); // threads per block
    dim3 nb_cost(w_valid, h_valid, nF_valid); // number of blocks
    size_t shared_memory_cost = (knn+1)*3*sizeof(int);
    update_cost<<<nb_cost, tpb_cost>>>(
            nVoxels,
            video_d,
            db_d,
            h, w, nF,
            h_db_valid, w_db_valid, nF_db_valid, 
            nC, knn,
            psz_space, psz_time,
            d_state,
            nnf_d,
            cost_d
    );
    cudaDeviceSynchronize();
    CudaCheckError();

    for (int iter = 0; iter < params_.propagation_iterations; ++iter) 
    {
        if(params_.verbosity > 0) {
            printf("  - iteration %d/%d\n", iter+1,params_.propagation_iterations);
        }

        // Jump-flood propagation
        for(int jump_flood_step = 1 ; jump_flood_step > 0; jump_flood_step /= 2) {
        // for(int jump_flood_step = params_.jump_flood_step ; jump_flood_step > 0; jump_flood_step /= 2) {
            if(params_.verbosity > 1) {
                printf("    jump flood with step %d\n", jump_flood_step);
            }

            // swap buffers (copy to tmp)
            cudaMemcpy(nnf_tmp_d, nnf_d, sz_nnf, cudaMemcpyDeviceToDevice);
            cudaMemcpy(cost_tmp_d, cost_d, sz_cost, cudaMemcpyDeviceToDevice);

            // flood, nnf_tmp is the nnf of the previous iteration, nnf is being updated
            dim3 tpb_propag(GPU_THREADS, 1, 1); // threads per block
            dim3 nb_propag(w_valid, h_valid, nF_valid); // number of blocks
            size_t shared_memory_propag = 
                psz_space*psz_space*psz_time*nC*sizeof(nnf_data_t)
                + 3*knn*sizeof(int)
                + knn*sizeof(nnf_data_t);
            pm_propag_kernel<<<nb_propag, tpb_propag, shared_memory_propag>>>(
                    nVoxels,
                    video_d,
                    db_d,
                    h, w, nF,
                    h_db_valid, w_db_valid, nF_db_valid, 
                    nC, knn,
                    psz_space, psz_time,
                    nnf_d, cost_d,
                    nnf_tmp_d, cost_tmp_d,
                    jump_flood_step
            );
            cudaDeviceSynchronize();
            CudaCheckError();
        } // jump flood

        // // Random sampling pass
        // pm_rs_kernel<<<MCP_GPU_GET_BLOCKS(nVoxels), MCP_GPU_NUM_THREADS>>>(
        //         nVoxels,
        //         video_d,
        //         db_d,
        //         h, w, nF,
        //         h_db_valid, w_db_valid, nF_db_valid, 
        //         nC, knn,
        //         psz_space, psz_time,
        //         d_state,
        //         best_match_d
        // );
        // cudaDeviceSynchronize();
    } // propagation iteration

    // Prepare host output buffers
    NNFieldOutput output(h,w,nF, knn);
    Video<int> &nnf          = output.nnf;
    Video<nnf_data_t> &error = output.error;

    // Copy buffers back to host
    cudaMemcpy(nnf.dataWriter(), nnf_d, sz_nnf, cudaMemcpyDeviceToHost);
    cudaMemcpy(error.dataWriter(), cost_d, sz_cost, cudaMemcpyDeviceToHost);

    // Cleanup
    cudaFree(d_state);   d_state    = nullptr;
    cudaFree(nnf_d);     nnf_d      = nullptr;
    cudaFree(nnf_tmp_d); nnf_tmp_d  = nullptr;
    cudaFree(cost_d);    cost_d     = nullptr;
    cudaFree(cost_tmp_d);cost_tmp_d = nullptr;
    cudaFree(video_d);   video_d    = nullptr;
    cudaFree(db_d);      db_d       = nullptr;

    return output;
}
