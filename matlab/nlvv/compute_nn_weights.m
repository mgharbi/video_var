function w = compute_nn_weights(w, knnf_params)
    % w has size (h,w,nFrames,kNN)

    % normalize uint8 -> float
    w = single(w)/(255*255*3);

    patch_count = knnf_params.patch_size_time*knnf_params.patch_size_space^2;

    w = exp(-0.5*w/patch_count/knnf_params.nn_bandwidth^2);

    % normalize across NN
    w = bsxfun(@rdivide, w, sum(w,4)+eps);

end % compute_nn_weights
