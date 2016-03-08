function new_video = update_regular_video(video_regular, video_warped, params)
% Updates the regular video and nearest-neighbors database, the
% warping field is fixed
% We compute a new_video that has more regularity (patch recurrence) than video_regular.
% new_video samples patches from video_regular and should be 'close' to video_warped.

new_video = video_regular;
for it_inner = 1:params.n_inner_iterations
    fprintf('  - Inner iteration %2d of %2d\n', it_inner, params.n_inner_iterations);

    % Get nearest neighbors
    t = tic;
    fprintf('    . %d-NN\t\t\t', params.knnf.knn);
    [nnf,dist] = knnfield(new_video, video_regular, params.knnf);
    fprintf('%.1fs.\n', toc(t));

    % get patch weights
    w = compute_nn_weights(dist, params.knnf);


    % coord(1) = 43;
    % coord(2) = round(size(nnf, 2)/2);
    % coord(3) = 6;
    % [vid, d, coords] = nn_gallery(new_video, video_regular,nnf,w,coord,params.knnf); 
    % space_time_scatter(params.plot3d, coord)
    % space_time_scatter(params.plot3d, coords,d, [1,.2, .2])
    % figure();
    % plot(d)
    % imwrite(video_slice(vid,1,round(size(vid,1)/2)), fullfile(params.debug.output, debug_path(sprintf('%d_nn_gallery.png', params.debug.outer_it))));

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

    % save_video(new_video, fullfile(params.debug.output, debug_path(sprintf('%d_%d_regular.mp4', params.debug.outer_it, it_inner))), false);

    nnfwarp = nnf_to_warp(nnf);
    nnfmap = zeros(size(nnf,1),(size(nnf,2)+1)*size(nnf,4)/3,size(nnf,3),3);
    % wmap = zeros(size(w,1), (size(w,2)+1)*size(w,4), size(w,3),1);
    for i = 1:size(nnf,4)/3
        nnfmap(:, (i-1)*(size(nnf,2)+1)+1:i*(size(nnf,2)+1)-1, :, :) = nnfwarp(:,:,:,3*(i-1)+1:3*i);
        % wmap(:, (i-1)*(size(w,2)+1)+1:i*(size(w,2)+1)-1, :, :) = w(:,:,:,i);
    end
    [space,time] = viz_warp(nnfmap);
    % save_video(space, fullfile(params.debug.output, debug_path(sprintf('%d_knnf_s.mp4', params.debug.outer_it))), true);
    % save_video(time, fullfile(params.debug.output, debug_path(sprintf('%d_knnf_t.mp4', params.debug.outer_it))), true);
    % wmap = normalize_video(wmap);

    sl = video_slice(new_video,2,round(size(new_video,2)/2));
    imwrite(sl, fullfile(params.debug.output, debug_path(sprintf('%d_%d_yt_slice.png',params.debug.outer_it, it_inner))));

    sl = video_slice(time,2,round(size(new_video,2)/2));
    imwrite(sl, fullfile(params.debug.output, debug_path(sprintf('%d_%d_yt_nnf_time.png',params.debug.outer_it, it_inner))));
    save_video(time, fullfile(params.debug.output, debug_path(sprintf('%d_knnf_t.mp4', params.debug.outer_it))), true);

    if it_inner > 1
        err = (new_video-video_prev).^2;
        err = mean(err(:));
        fprintf('    . residual error:\t%g\n', err);
    else
        video_prev = new_video;
    end
end


% nnfwarp = nnf_to_warp(nnf);
% nnfmap = zeros(size(nnf,1),(size(nnf,2)+1)*size(nnf,4)/3,size(nnf,3),3);
% wmap = zeros(size(w,1), (size(w,2)+1)*size(w,4), size(w,3),1);
% for i = 1:size(nnf,4)/3
%     nnfmap(:, (i-1)*(size(nnf,2)+1)+1:i*(size(nnf,2)+1)-1, :, :) = nnfwarp(:,:,:,3*(i-1)+1:3*i);
%     wmap(:, (i-1)*(size(w,2)+1)+1:i*(size(w,2)+1)-1, :, :) = w(:,:,:,i);
% end
% wmap = normalize_video(wmap);
% save_video(wmap, fullfile(params.debug.output, debug_path(sprintf('%d_wmap.mp4', params.debug.outer_it))), true);

% [space,time] = viz_warp(nnfmap);
% save_video(space, fullfile(params.debug.output, debug_path(sprintf('%d_knnf_s.mp4', params.debug.outer_it))), true);
% save_video(time, fullfile(params.debug.output, debug_path(sprintf('%d_knnf_t.mp4', params.debug.outer_it))), true);


end % update_regular_video
