function params = knnf_params()
    params                        = struct();
    params.propagation_iterations = 1;
    params.patch_size_space       = 5;
    params.patch_size_time        = 5;
    params.knn                    = 3; % default 20
    params.threads                = 16;

    params.verbosity              = 0;

    params.nn_bandwidth = 0.1;

end % knnf_params
