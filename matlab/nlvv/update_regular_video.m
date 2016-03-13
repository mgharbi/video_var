function [new_video, nnf] = update_regular_video(video_regular, video_warped, params)
% -------------------------------------------------------------------------
% Updates the regular video and nearest-neighbors database, the
% warping field is fixed
% We compute a new_regular_video that has more regularity (patch recurrence) than video_regular.
% new_regular_video samples patches from video_regular and should be 'close' to video_warped.
% -------------------------------------------------------------------------

new_regular_video = video_regular;

for it_inner = 1:params.n_inner_iterations
    fprintf('  - Inner iteration %2d of %2d\n', it_inner, params.n_inner_iterations);

    % Get nearest neighbors
    t = tic;
    fprintf('    . %d-NN\t\t\t', params.knnf.knn);
    [nnf,dist] = knnfield(new_regular_video, video_regular, params.knnf);
    fprintf('%.1fs.\n', toc(t));

    % get patch weights
    w = compute_nn_weights(dist, params.knnf);

    % Assemble new video by sampling patches from the previous regular video
    t = tic;
    fprintf('    . Assemble regular video\t');
    new_regular_video = reconstruct_from_knnf(video_regular, nnf, w, params.knnf);
    fprintf('%.1fs.\n', toc(t));

    % Fuse new video and warped video to produce the final result
    warped_weight = params.data_term_weight*...
        repmat(1./robust_cost(video_warped, new_regular_video), [1,1,1,3]);
    new_regular_video_weight = 1/params.knnf.nn_bandwidth^2;
    new_regular_video = (warped_weight.*single(video_warped) + new_regular_video_weight*single(new_regular_video))./ (warped_weight+new_regular_video_weight);
    new_regular_video = uint8(new_regular_video);

    fprintf('fusing new regular. w_warped = %f | w_new_regular = %f\n', mean(warped_weight(:)), 1/params.knnf.nn_bandwidth^2);
end

end % update_regular_video
