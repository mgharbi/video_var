function params   = nlvv_params()
    params        = struct();
    % params.knn    = knnf_params();
    % params.stwarp = stwarp_params();

    params.n_outer_iterations  = 1;
    params.n_inner_iterations  = 1;
    params.patch_size_spatial  = 15;
    params.patch_size_temporal = 5;
    params.nn_count            = 10;

end % knnf_params
