function updated_video = update_regular_video(video_regular, video_warped, params)
% Updates the regular video and nearest-neighbors database, the
% warping field is fixed

updated_video = video_regular;
nn_field = [];
for it_inner = 1:params.n_inner_iterations
    % Get nearest neighbors
    fprintf('Finding %d-NN from the regular video\n', params.nn_count);

    % Generate new video from nearest neighbor patches
    fprintf('\n', params.nn_count);

    % Fuse new video and warped video to produce the final result
end

end % update_regular_video
