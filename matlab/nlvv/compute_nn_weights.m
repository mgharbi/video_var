function w = compute_nn_weights(w, knnf_params)
% -------------------------------------------------------------------------
% The NNF codes outputs square pixel difference, we normalize these distances and turn
% them into properly normalized probabilistic weights.
% -------------------------------------------------------------------------
    range = 1; % 255 for uint8, 1 for float;

    % w has size (h,w,nFrames,kNN)
    patch_count = knnf_params.patch_size_time*knnf_params.patch_size_space^2;

    % normalize uint8 -> float
    w = single(w)/(range*range*3)/patch_count;

    w = exp(-0.5*w/knnf_params.nn_bandwidth^2);

    % normalize across NN
    w = bsxfun(@rdivide, w, sum(w,4)+eps);

end % compute_nn_weights
