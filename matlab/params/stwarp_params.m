function params = stwarp_params()
    params = struct();

    reg_scaler = 1e-2;
    params.reg_spatial_uv  = 1*reg_scaler;
    params.reg_temporal_uv = 1*reg_scaler;
    params.reg_spatial_w   = 1*reg_scaler;
    params.reg_temporal_w  = 1*reg_scaler;

    params.use_color     = true;
    params.use_gradients = true; % Use gradient-matching in addition to pixel values

    params.limit_update = false;

    params.median_filter_size = 5;
    params.use_advanced_median = false; % Edge-Aware median filtering

    params.pyramid_spacing  = 1.25;
    params.min_pyramid_size = 8; % resolution of the smalles pyramid level
    params.pyramid_levels   = -1; % -1 for auto-levels

    params.solver_iterations = 50;
    params.warping_iterations = 5;

    params.verbosity = 0;
end % knnf_params
