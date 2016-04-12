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
