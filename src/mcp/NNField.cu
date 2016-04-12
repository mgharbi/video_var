#include <algorithm>
#include <cassert>
#include <cfloat>

#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>
#include <thrust/execution_policy.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>

#include "cuda/utils.h"
#include "mcp/NNField.hpp"

// ---- GPU datastructures ---------------------------------------

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
        // MatchGPU* bestmatches
        int* nnf,
        int* cost
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

        // Copy PRNG state to local memory
        curandState state = rng_state[voxel];

        // // Copy patch
        // for (int t = 0; t < psz_time; ++t) 
        // for (int x = 0; x < psz_space; ++x) 
        // for (int y = 0; y < psz_space; ++y) 
        // {
        //     int vx_video = y+vx_y + h*( x+vx_x + w*(t+vx_t));
        //     int vx_db = y+y_db + h*( x+x_db + w*(t+t_db));
        //     for( int c = 0 ; c < nC ; c++ ){
        //         nnf_data_t d = (float)video[vx_video+c*h*w*nF] - (float)db[vx_db+c*h*w*nF];
        //         cost += d*d;
        //     }
        // }
        //
        // __shared__ nnf_data_t patch[11*11*11*3];
        // nnf_data_t *patch = new nnf_data_t[psz_time*psz_space*psz_space*nC];
        for (int k = 0; k < knn; ++k) {
            int t_db = curand_uniform(&state) * nF_db_valid;
            int x_db = curand_uniform(&state) * w_db_valid;
            int y_db = curand_uniform(&state) * h_db_valid;
            nnf[voxel*3*(knn+1) + k + 0] = x_db;
            nnf[voxel*3*(knn+1) + k + 1] = y_db;
            nnf[voxel*3*(knn+1) + k + 2] = t_db;
            // cost[voxel*(knn+1) + k] = 
            //     d_patch_cost(video, db, h, w, nF, nC,
            //         vx_y,vx_x,vx_t, y_db,x_db,t_db,
            //         psz_space, psz_time);

            // bestmatches[voxel*(knn+1) + k].x = x_db;
            // bestmatches[voxel*(knn+1) + k].y = y_db;
            // bestmatches[voxel*(knn+1) + k].t = t_db;
            // bestmatches[voxel*(knn+1) + k].cost = 
            //     d_patch_cost(video, db, h, w, nF, nC,
            //         vx_y,vx_x,vx_t, y_db,x_db,t_db,
            //         psz_space, psz_time);

            float c = 0;
            for (int t = 0; t < psz_time; ++t) 
            for (int x = 0; x < psz_space; ++x) 
            for (int y = 0; y < psz_space; ++y) 
            {
                // int vx_video = y+vx_y + h*( x+vx_x + w*(t+vx_t));
                // int vx_patch = y+vx_y + psz_space*( x + vx_x + psz_space*(t+vx_t));
                int vx_db = y+y_db + h*( x+x_db + w*(t+t_db));
                for( int c = 0 ; c < nC ; c++ ){
                    // nnf_data_t d = patch[0]-(float)db[vx_db+c*h*w*nF];
                    // nnf_data_t d = video[vx_video+c*h*w*nF] - db[vx_db+c*h*w*nF];
                    c += d*d;
                }
            }
            cost[voxel*(knn+1) + k]= c;
        } // knn loop
        // delete[] patch;

        // thrust::sort(thrust::seq, bestmatches+voxel*(knn+1),bestmatches+voxel*(knn+1) +knn, MatchGPUCompare());

        // Copy back PRNG state to global memory
        rng_state[voxel] = state;
    } // buffer px cudaloop
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
        MatchGPU* bestmatches,
        MatchGPU* bestmatches_tmp,
        int jump
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

        // Fetch best (from tmp: we want to use the previous iteration to avoid race cditions)
        MatchGPU *current_best = bestmatches + (knn+1)*voxel; // 1 extra for insertion/deletion

        // Propagate x
        if(vx_x - jump >= 0) {
            int voxel_p = voxel - jump*h;
            for (int k = 0; k < knn; ++k) { // get k next neighbors
                int x_p = bestmatches_tmp[voxel_p*(knn+1) + k].x + jump;
                int y_p = bestmatches_tmp[voxel_p*(knn+1) + k].y;
                int t_p = bestmatches_tmp[voxel_p*(knn+1) + k].t;
                if(x_p < w_db_valid) {
                    d_improve_knn(video, db, h, w, nF, nC, knn, psz_space, psz_time,
                        vx_y, vx_x, vx_t, y_p, x_p, t_p, current_best);
                }
            }
        }
        if(vx_x + jump < w) {
            int voxel_p = voxel + jump*h;
            for (int k = 0; k < knn; ++k) { // get k next neighbors
                int x_p = bestmatches_tmp[voxel_p*(knn+1) + k].x - jump;
                int y_p = bestmatches_tmp[voxel_p*(knn+1) + k].y;
                int t_p = bestmatches_tmp[voxel_p*(knn+1) + k].t;
                if(x_p >= 0 && x_p < w_db_valid) {
                    d_improve_knn(video, db, h, w, nF, nC, knn, psz_space, psz_time,
                        vx_y, vx_x, vx_t, y_p, x_p, t_p, current_best);
                }
            }
        }

        // Propagate y
        if(vx_y - jump >= 0) {
            int voxel_p = voxel - jump;
            for (int k = 0; k < knn; ++k) { // get k next neighbors
                int x_p = bestmatches_tmp[voxel_p*(knn+1) + k].x;
                int y_p = bestmatches_tmp[voxel_p*(knn+1) + k].y + jump;
                int t_p = bestmatches_tmp[voxel_p*(knn+1) + k].t;
                if(y_p < h_db_valid) {
                    d_improve_knn(video, db, h, w, nF, nC, knn, psz_space, psz_time,
                        vx_y, vx_x, vx_t, y_p, x_p, t_p, current_best);
                }
            }
        }
        if(vx_y + jump < h) {
            int voxel_p = voxel + jump;
            for (int k = 0; k < knn; ++k) { // get k next neighbors
                int x_p = bestmatches_tmp[voxel_p*(knn+1) + k].x;
                int y_p = bestmatches_tmp[voxel_p*(knn+1) + k].y - jump;
                int t_p = bestmatches_tmp[voxel_p*(knn+1) + k].t;
                if(y_p >= 0 && y_p < h_db_valid) {
                    d_improve_knn(video, db, h, w, nF, nC, knn, psz_space, psz_time,
                        vx_y, vx_x, vx_t, y_p, x_p, t_p, current_best);
                }
            }
        }

        // Propagate t
        if(vx_t - jump >= 0) {
            int voxel_p = voxel - h*w*jump;
            for (int k = 0; k < knn; ++k) { // get k next neighbors
                int x_p = bestmatches_tmp[voxel_p*(knn+1) + k].x;
                int y_p = bestmatches_tmp[voxel_p*(knn+1) + k].y;
                int t_p = bestmatches_tmp[voxel_p*(knn+1) + k].t + jump;
                if(t_p < nF_db_valid) {
                    d_improve_knn(video, db, h, w, nF, nC, knn, psz_space, psz_time,
                        vx_y, vx_x, vx_t, y_p, x_p, t_p, current_best);
                }
            }
        }
        if(vx_t + jump < nF) {
            int voxel_p = voxel + h*w*jump;
            for (int k = 0; k < knn; ++k) { // get k next neighbors
                int x_p = bestmatches_tmp[voxel_p*(knn+1) + k].x;
                int y_p = bestmatches_tmp[voxel_p*(knn+1) + k].y;
                int t_p = bestmatches_tmp[voxel_p*(knn+1) + k].t - jump;
                if(t_p >= 0 && t_p < nF_db_valid) {
                    d_improve_knn(video, db, h, w, nF, nC, knn, psz_space, psz_time,
                        vx_y, vx_x, vx_t, y_p, x_p, t_p, current_best);
                }
            }
        }
    } // buffer px cudaloop
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

    // Allocate GPU output buffers
    MatchGPU* best_match_d     = nullptr;
    MatchGPU* best_match_tmp_d = nullptr;
    int sz_bestmatches         = h*w*nF*(knn+1)*sizeof(MatchGPU);
    cudaMalloc((void**) &best_match_d, sz_bestmatches);
    cudaMalloc((void**) &best_match_tmp_d, sz_bestmatches);

    int* nnf_d     = nullptr;
    int* cost_d     = nullptr;
    int sz_nnf         = h*w*nF*3*(knn+1)*sizeof(int);
    int sz_cost         = h*w*nF*1*(knn+1)*sizeof(int);
    cudaMalloc((void**) &nnf_d, sz_nnf);
    cudaMalloc((void**) &cost_d, sz_cost);

    // init
    if(params_.verbosity > 0) {
        printf("+ NNF (gpu) initialization with size %dx%dx%d, ",h,w,nF);
        printf("patch size %dx%d\n",psz_space,psz_time);
    };

    // Prepare random number generator
    curandState *d_state;
    cudaMalloc(&d_state, nVoxels*sizeof(curandState));
    setup_rng_kernel<<<MCP_GPU_GET_BLOCKS(nVoxels), MCP_GPU_NUM_THREADS>>>(nVoxels, d_state);


    const int TILE_W = 16;
    const int TILE_H = 16;
    dim3 threadsPerBlock(TILE_W,TILE_H);
    dim3 numBlocks(w/threadsPerBlock.x, h/threadsPerBlock.y);
    initialize_kernel<<<numBlocks, threadsPerBlock>>>(
    // initialize_kernel<<<MCP_GPU_GET_BLOCKS(nVoxels), MCP_GPU_NUM_THREADS>>>(
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
            // best_match_d
    );
    // cudaDeviceSynchronize();

    // for (int iter = 0; iter < params_.propagation_iterations; ++iter) 
    // {
    //     if(params_.verbosity > 0) {
    //         printf("  - iteration %d/%d\n", iter+1,params_.propagation_iterations);
    //     }
    //
    //     // Jump-flood propagation
    //     for(int jump_flood_step = params_.jump_flood_step ; jump_flood_step > 0; jump_flood_step /= 2) {
    //         if(params_.verbosity > 1) {
    //             printf("    jump flood with step %d\n", jump_flood_step);
    //         }
    //
    //         // swap buffers (copy to tmp)
    //         cudaMemcpy(best_match_tmp_d, best_match_d, sz_bestmatches, cudaMemcpyDeviceToDevice);
    //
    //         // flood, nnf_tmp is the nnf of the previous iteration, nnf is being updated
    //         pm_propag_kernel<<<MCP_GPU_GET_BLOCKS(nVoxels), MCP_GPU_NUM_THREADS>>>(
    //                 nVoxels,
    //                 video_d,
    //                 db_d,
    //                 h, w, nF,
    //                 h_db_valid, w_db_valid, nF_db_valid, 
    //                 nC, knn,
    //                 psz_space, psz_time,
    //                 best_match_d,
    //                 best_match_tmp_d,
    //                 jump_flood_step
    //         );
    //         cudaDeviceSynchronize();
    //     } // jump flood
    //
    //     // Random sampling pass
    //     pm_rs_kernel<<<MCP_GPU_GET_BLOCKS(nVoxels), MCP_GPU_NUM_THREADS>>>(
    //             nVoxels,
    //             video_d,
    //             db_d,
    //             h, w, nF,
    //             h_db_valid, w_db_valid, nF_db_valid, 
    //             nC, knn,
    //             psz_space, psz_time,
    //             d_state,
    //             best_match_d
    //     );
    //     cudaDeviceSynchronize();
    // } // propagation iteration

    // Prepare host output buffers
    NNFieldOutput output(h,w,nF, knn);
    Video<int> &nnf          = output.nnf;
    Video<nnf_data_t> &error = output.error;

    // Copy buffers back to host
    MatchGPU *best_match = new MatchGPU[h*w*nF*(knn+1)]();
    cudaMemcpy(best_match, best_match_d, sz_bestmatches, cudaMemcpyDeviceToHost);
    for (int voxel = 0; voxel < h*w*nF; ++voxel)
    {
        for (int k = 0; k < knn; ++k)
        {
            MatchGPU m = best_match[voxel*(knn+1) + k];
            nnf.setAt(m.x, voxel, 0, 0, 3*k+0 );
            nnf.setAt(m.y, voxel, 0, 0, 3*k+1 );
            nnf.setAt(m.t, voxel, 0, 0, 3*k+2);
            error.setAt(m.cost, voxel, 0, 0, k);
        }
    }
    delete[] best_match;

    // Cleanup
    cudaFree(d_state);          d_state          = nullptr;
    cudaFree(best_match_d);     best_match_d     = nullptr;
    cudaFree(best_match_tmp_d); best_match_tmp_d = nullptr;
    cudaFree(video_d);          video_d          = nullptr;
    cudaFree(db_d);             db_d             = nullptr;

    return output;
}
