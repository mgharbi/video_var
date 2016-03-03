function params = knnf_params()
    params                        = struct();
    params.propagation_iterations = 3;
    params.patch_size_space       = 15;
    params.patch_size_time        = 15;
    params.knn                    = 20; % default 20
    params.threads                = 16;

    params.verbosity              = 0;

    params.nn_bandwidth = .1;

end % knnf_params
