function updated_video = update_regular_video(video_regular, video_warped, params)
% Updates the regular video and nearest-neighbors database, the
% warping field is fixed

updated_video = video_regular;
nn_field = [];
for it_inner = 1:params.n_inner_iterations
    % Get nearest neighbors
    fprintf('  - %d-NN\t\t\n', params.knnf.knn);
    t = tic;
    [nnf,dist] = knnfield(updated_video, video_regular, params.knnf);
    fprintf('%.1fs.\n', toc(t));

    % get patch weights
    w = compute_nn_weights(dist, params.knnf);

    % assemble new image by sampling patches from the previous regular image
    new_video = reconstruct_from_knnf(video_regular, nnf, w, params.knnf);

    keyboard

    % Generate new video from nearest neighbor patches
    % fprintf('\n', params.nn_count);

    % Fuse new video and warped video to produce the final result
end

end % update_regular_video
