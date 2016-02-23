#include "mcp/NNField.hpp"
#include <algorithm>
#if defined(_OPENMP)
#include <omp.h>
#endif

NNField::~NNField() {
}

float NNField::getPatchCost(const IVideo &A,const IVideo &B, int y_a, int x_a, int t_a, int y_b, int x_b, int t_b) 
{
    float ans = 0;
    const unsigned char* pA = A.dataReader();
    const unsigned char* pB = B.dataReader();

    int h       = A.getHeight();
    int w       = A.getWidth();
    int nVoxels = A.voxelCount();

    for (int t = 0; t < params_.patch_size_time; ++t) 
    for (int x = 0; x < params_.patch_size_space; ++x) 
    for (int y = 0; y < params_.patch_size_space; ++y) 
    {
        int indexA = y+y_a + h*( x+x_a + w*(t+t_a));
        int indexB = y+y_b + h*( x+x_b + w*(t+t_b));
        for( int l = 0 ; l < A.channelCount() ; l++ ){
            float c = (float)pA[indexA+l*nVoxels] - (float)pB[indexB+l*nVoxels];
            ans += c*c;
        }
    }
    return ans;
}

void NNField::improve_guess(const IVideo &A,const IVideo &B,int y_a, int x_a, int t_a,
    int &y_best, int &x_best, int &t_best, float& cost,
    int y_p, int x_p, int t_p)
{
    float d = getPatchCost(A,B, y_a,x_a,t_a,y_p,x_p,t_p);
    if( d < cost) {
        cost = d;
        y_best = y_p;
        x_best = x_p;
        t_best = t_p;
    }
}

void NNField::improve_knn(const IVideo &A,const IVideo &B,int y_a, int x_a, int t_a,
    vector<Match> &current_best,
    PatchCoordHashMap &all_matches,
    int y_p, int x_p, int t_p)
{
    // if current pos in hash map, skip
    if(all_matches.count(PatchCoord(x_p,y_p,t_p))) {
        return;
    }

    // compute distance
    float candidate_dist = getPatchCost(A,B, y_a,x_a,t_a,y_p,x_p,t_p);

    float err = std::get<0>(current_best[0]);

    if( candidate_dist < err) { // we have a better match
        std::pop_heap(current_best.begin(), current_best.end());
        current_best.pop_back();

        current_best.push_back(Match(candidate_dist,x_p,y_p,t_p));
        std::push_heap(current_best.begin(), current_best.end());
    }
}

Video<int> NNField::compute() {

    int h       = video_->getHeight(); int h_valid = h - params_.patch_size_space+1;
    int w       = video_->getWidth(); int w_valid = w - params_.patch_size_space+1;
    int nF      = video_->frameCount(); int nF_valid = nF - params_.patch_size_time+1;
    int nVoxels = video_->voxelCount();

    // So that patches don't read out of database
    int h_db  = database_->getHeight(); int h_db_valid = h_db  - params_.patch_size_space + 1;
    int w_db  = database_->getWidth(); int w_db_valid = w_db  - params_.patch_size_space + 1;
    int nF_db  = database_->frameCount(); int nF_db_valid = nF_db  - params_.patch_size_time + 1;

    Video<int> nnf(h, w,  nF, 3*params_.knn);
    Video<float> nnf_dist(h, w,  nF, params_.knn);
    
    nn_offset_ = 3*nVoxels;

    int n_threads = 1;
    #if defined(_OPENMP)
        n_threads = params_.threads;
    #endif
    cout << "Using OpenMP with " << n_threads << " threads" << endl;

    // init
    printf("+ NNF initialization with size %dx%dx%d, ",h,w,nF);
    printf("patch size %dx%d\n",params_.patch_size_space,params_.patch_size_time);
    #pragma omp parallel for num_threads(n_threads)
    for (int t = 0; t < nF - params_.patch_size_time  + 1; ++t) 
    {
        int* pNNF    = nnf.dataWriter();
        float* pCost = nnf_dist.dataWriter();
        for (int x = 0; x < w  - params_.patch_size_space + 1; ++x)
        {
            for (int y = 0; y < h  - params_.patch_size_space + 1; ++y) 
            {
                int index = y + h*(x + w*t);
                for (int k = 0 ; k < params_.knn; ++ k) {
                    int t_db               = rand() % nF_db_valid;
                    int x_db               = rand() % w_db_valid;
                    int y_db               = rand() % h_db_valid;
                    pNNF[index + 0*nVoxels + k*nn_offset_] = x_db;
                    pNNF[index + 1*nVoxels + k*nn_offset_] = y_db;
                    pNNF[index + 2*nVoxels + k*nn_offset_] = t_db;
                    pCost[index + k*nVoxels] = getPatchCost(*video_, *database_, y,x,t,y_db,x_db,t_db);
                }
            } // y loop
        } // x loop
    } // t loop

    for (int iter = 0; iter < params_.propagation_iterations; ++iter) 
    {
        printf("  - iteration %d/%d\n", iter+1,params_.propagation_iterations);
        #pragma omp parallel num_threads(n_threads)
        {
            int* pNNF    = nnf.dataWriter();
            float* pCost = nnf_dist.dataWriter();
            int tmin = 0;
            int tmax = nF_valid;
            int xmin = 0;
            int xmax = w_valid;
            int ymin = 0;
            int ymax = h_valid;
            #if defined(_OPENMP)
                int ithread = omp_get_thread_num();
                // Split the 3d volume along the t-axis
                if(nF_valid > omp_get_num_threads() ) {
                    int tile_nf = nF_valid / omp_get_num_threads();
                    tmin = ithread*tile_nf;
                    tmax = min((ithread+1)*tile_nf,nF_valid);
                }else {
                    int tile_w = w_valid / omp_get_num_threads();
                    xmin = ithread*tile_w;
                    xmax = min((ithread+1)*tile_w,w_valid);
                }
            #endif

            // Travel direction, reverse at every other iteration
            int tstart = tmin, tend = tmax, tchange = 1;
            int xstart = xmin, xend = xmax, xchange = 1;
            int ystart = ymin, yend = ymax, ychange = 1;
            if (iter % 2 == 1) {
                tstart = tmax-1; tend = tmin-1; tchange = -1;
                xstart = xmax-1; xend = xmin-1; xchange = -1;
                ystart = ymax-1; yend = ymin-1; ychange = -1;
            }


            // Loop through all patches in the video
            for (int y = ystart; y != yend; y += ychange) 
            for (int t = tstart; t != tend; t += tchange) 
            for (int x = xstart; x != xend; x += xchange) 
            { 
                int index = y + h*(x+w*t);

                // get current best k-nn of the patch under consideration
                vector<Match> current_best;
                PatchCoordHashMap all_matches;
                current_best.reserve(params_.knn);
                for (int k = 0; k < params_.knn; ++k) {
                    int x_best    = pNNF[index + 0*nVoxels + k*nn_offset_];
                    int y_best    = pNNF[index + 1*nVoxels + k*nn_offset_];
                    int t_best    = pNNF[index + 2*nVoxels + k*nn_offset_];
                    float best_cost = pCost[index + k*nVoxels];
                    current_best.push_back(Match(best_cost, x_best,y_best,t_best));
                    all_matches.emplace(PatchCoord(x_best,y_best,t_best),true);
                }
                make_heap(current_best.begin(), current_best.end());

                // propagate all k-nn to the next location

                // Propagate x
                if ( x - xchange > -1 && x - xchange <  w_valid) {
                    int index_prev = index - h*xchange;
                    int x_p = pNNF[index_prev + 0*nVoxels] + xchange;
                    int y_p = pNNF[index_prev + 1*nVoxels];
                    int t_p = pNNF[index_prev + 2*nVoxels];
                    if ( x_p> -1 && x_p <  w_db_valid) {
                        improve_knn(*video_,*database_,y, x, t,
                                current_best,
                                all_matches,
                                y_p, x_p, t_p);
                    }
                }

                // Propagate y
                if ( y - ychange > -1 && y - ychange <  h_valid) {
                    int index_prev = index - ychange;
                    int x_p = pNNF[index_prev + 0*nVoxels];
                    int y_p = pNNF[index_prev + 1*nVoxels] + ychange;
                    int t_p = pNNF[index_prev + 2*nVoxels];
                    if ( y_p> -1 && y_p <  h_db_valid) {
                        improve_knn(*video_,*database_,y, x, t,
                                current_best,
                                all_matches,
                                y_p, x_p, t_p);
                    }
                }

                // Propagate t
                if ( t - tchange > -1 && t - tchange <  nF_valid) {
                    int index_prev = index - tchange*h*w;
                    int x_p = pNNF[index_prev + 0*nVoxels];
                    int y_p = pNNF[index_prev + 1*nVoxels];
                    int t_p = pNNF[index_prev + 2*nVoxels] + tchange;
                    if ( t_p> -1 && t_p <  nF_db_valid) {
                        improve_knn(*video_,*database_,y, x, t,
                                current_best,
                                all_matches,
                                y_p, x_p, t_p);
                    }
                }

                // Random search new guesses
                int rs_start = numeric_limits<int>::max();
                int rt_start = numeric_limits<int>::max();

                if (rs_start > max(w, h)) { rs_start = max(w, h); }
                if (rt_start > nF) { rt_start = nF; }

                // int mag_time = rt_start;
                for (int mag = rs_start; mag >= 1; mag /= 2)
                {
                    int mag_time = mag;
                    // if(mag_time >= 1) {
                    //     mag_time /= 2;
                    // }

                    for(int k = 0; k < params_.knn; ++k)
                    {

                        /* Sampling window */
                        int x_best = std::get<1>(current_best[k]);
                        int y_best = std::get<2>(current_best[k]);
                        int t_best = std::get<3>(current_best[k]);

                        int y_min = max(y_best-mag, 0);
                        int x_min = max(x_best-mag, 0);
                        int t_min = max(t_best-mag_time, 0); 
                        int y_max = min(y_best+mag+1,h_db_valid);
                        int x_max = min(x_best+mag+1,w_db_valid);
                        int t_max = min(t_best+mag_time+1,nF_db_valid);

                        // New random proposal from the region
                        int y_p    = y_min + rand() % (y_max-y_min);
                        int x_p    = x_min + rand() % (x_max-x_min);
                        int t_p    = t_min + rand() % (t_max-t_min);

                        improve_knn(*video_,*database_,y, x, t,
                                current_best,
                                all_matches,
                                y_p, x_p, t_p);

                    }
                }
                std::sort(current_best.begin(), current_best.end());
                for (int k = 0 ; k < params_.knn; ++ k) {
                    pNNF[index + 0*nVoxels + k*nn_offset_] = std::get<1>(current_best[k]);
                    pNNF[index + 1*nVoxels + k*nn_offset_] = std::get<2>(current_best[k]);
                    pNNF[index + 2*nVoxels + k*nn_offset_] = std::get<3>(current_best[k]);
                    pCost[index + k*nVoxels] = std::get<0>(current_best[k]);
                }
            } // patches loop

        } // parallel 

        float* pCost = nnf_dist.dataWriter();
        vector<float> avg_cost(params_.knn);
        for (int k = 0; k < params_.knn; ++k)
            for (int i = 0; i < nVoxels; ++i)
            {
                avg_cost[k] += pCost[i+k*nVoxels];
            }
        for (int k = 0; k < params_.knn; ++k)
        {
            avg_cost[k] /= nVoxels;
            cout << "    match cost [" << k << "] : " << avg_cost[k] << endl;;
        }

    } // propagation iteration



    return nnf;
}
