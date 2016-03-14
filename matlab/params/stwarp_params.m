function params = stwarp_params()
    params = struct();

    reg_scaler = 1e-2;
    params.reg_spatial_uv  = 1*reg_scaler;
    params.reg_temporal_uv = 1*reg_scaler;
    params.reg_spatial_w   = 1*reg_scaler;
    params.reg_temporal_w  = 1*reg_scaler;

    params.verbosity = 0;
end % knnf_params
