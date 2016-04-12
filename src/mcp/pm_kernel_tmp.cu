    // END
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
