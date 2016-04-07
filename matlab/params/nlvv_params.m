function params = nlvv_params()
    params        = struct();
    params.knnf   = knnf_params();
    params.stwarp = stwarp_params();

    params.n_outer_iterations  = 10; 
    params.n_inner_iterations  = 5; % updating the regular video

    params.data_term_weight = .2; % lambda in Tali's paper

    params.pyramid_ratio  = 0.75;
    params.scale_min  = 4;
    params.scale_max  = 9;
end % knnf_params
