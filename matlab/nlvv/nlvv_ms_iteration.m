function res = nlvv_ms_iteration(video,params,video_regular,warp_field)
    % Initialization
    [h, w, nframes, nchans] = size(video);
    if isempty(warp_field) 
        display('* Initialize identity warping field');
        warp_field = zeros(h,w,nframes,3);
        video_warped = video; % No need to warp if the transformation is identity
    else
        video_warped = backward_warp(video, warp_field);
    end


    if isempty(video_regular) 
        display('* Initialize regular video');
        video_regular = video_warped;
    end

    res.warp_field = warp_field;

    for it_outer = 1:params.n_outer_iterations
        fprintf('* Outer iteration %2d of %2d\n', it_outer, params.n_outer_iterations);

        params.debug.outer_it = it_outer;

        % Regularize video
        video_regular = update_regular_video(video_regular, video_warped, params);

        % warp video towards towards video_regular
        t = tic;
        fprintf('  - Compute warp\t\t');
        warp_field = stwarp(video, video_regular,params.stwarp);
        fprintf('%.1fs.\n', toc(t));

        [space,time] = viz_warp(warp_field);
        save_video(space, fullfile(params.debug.output, debug_path(sprintf('%d_warp_s.mp4', params.debug.outer_it))), true);
        save_video(time, fullfile(params.debug.output, debug_path(sprintf('%d_warp_t.mp4', params.debug.outer_it))), true);

        % Fix boundaries and inversion

        video_warped = backward_warp(video_warped, warp_field);
        save_video(video_warped, fullfile(params.debug.output, debug_path(sprintf('%d_warped.mp4', params.debug.outer_it))));
    end

    res.video_regular = video_regular;
    res.warp_field = warp_field;
end % nlvv_ms_iteration
