function new_video = update_regular_video(video_regular, video_warped, params)
% Updates the regular video and nearest-neighbors database, the
% warping field is fixed

new_video = video_regular;
nn_field = [];
for it_inner = 1:params.n_inner_iterations
    fprintf('  - Inner iteration %2d of %2d\n', it_inner, params.n_inner_iterations);
    % Get nearest neighbors
    t = tic;
    fprintf('    . %d-NN\t\t\t', params.knnf.knn);
    [nnf,dist] = knnfield(new_video, video_regular, params.knnf);
    fprintf('%.1fs.\n', toc(t));

    [space,time] = viz_warp(single(nnf(:,:,:,1:3)));
    save_video(space, fullfile(params.debug.output, debug_path(sprintf('%d_knnf_s.mp4', params.debug.outer_it))), true);
    save_video(time, fullfile(params.debug.output, debug_path(sprintf('%d_knnf_t.mp4', params.debug.outer_it))), true);

    % get patch weights
    w = compute_nn_weights(dist, params.knnf);

    wmap = normalize_video(w(:,:,:,1));
    save_video(wmap, fullfile(params.debug.output, debug_path(sprintf('%d_wmap.mp4', params.debug.outer_it))), true);

    % Assemble new video by sampling patches from the previous regular video
    t = tic;
    fprintf('    . Assemble regular video\t');
    new_video = reconstruct_from_knnf(video_regular, nnf, w, params.knnf);
    fprintf('%.1fs.\n', toc(t));


    % Fuse new video and warped video to produce the final result
    warped_weight = params.data_term_weight*...
        repmat(1./robust_cost(video_warped, new_video), [1,1,1,3]);
    new_video_weight = 1/params.knnf.nn_bandwidth^2;
    new_video = (warped_weight.*single(video_warped) + new_video_weight*single(new_video))./ (warped_weight+new_video_weight);
    new_video = uint8(new_video);

    save_video(new_video, fullfile(params.debug.output, debug_path(sprintf('%d_regular.mp4', params.debug.outer_it))), false);
end

end % update_regular_video
