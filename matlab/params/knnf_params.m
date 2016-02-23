function params = knnf_params()
    params                        = struct();
    params.propagation_iterations = 3;
    params.patch_size_space       = 5;
    params.patch_size_time        = 5;
    params.knn                    = 1;
    params.threads                = 16;
end % knnf_params
