function params = nlvv_params()
    params        = struct();
    params.knnf   = knnf_params();
    params.stwarp = stwarp_params();

    params.n_outer_iterations  = 1; % default 10
    params.n_inner_iterations  = 1;

end % knnf_params
