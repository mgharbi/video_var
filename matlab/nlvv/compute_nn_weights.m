function w = compute_nn_weights(w, knnf_params)
    % w has size (h,w,nFrames,kNN)

    patch_count = knnf_params.patch_size_time*knnf_params.patch_size_space^2;

    % normalize uint8 -> float
    w = single(w)/(255*255*3)/patch_count;

    w = exp(-0.5*w/knnf_params.nn_bandwidth^2);

    % normalize across NN
    w = bsxfun(@rdivide, w, sum(w,4)+eps);

    % ww = w(:,:,:,1); ww = ww(:);
    % ww2 = w(:,:,:,2); ww2 = ww2(:);
    % [ww,i] =sort(ww);
    % ww2 = ww2(i);
    % plot(sort(ww(:)))
    % hold on
    % plot((ww2(:)))
    % w = single(w);

end % compute_nn_weights
