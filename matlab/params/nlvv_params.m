function params = nlvv_params()
    params        = struct();
    params.knnf   = knnf_params();
    params.stwarp = stwarp_params();

    params.n_outer_iterations  = 10; % default 10
    params.n_inner_iterations  = 3;

    params.data_term_weight = 20; % lambda in the paper
end % knnf_params
