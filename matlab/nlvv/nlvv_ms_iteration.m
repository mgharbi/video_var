function res = nlvv_ms_iteration(video,params,video_regular,warp_field)
    % Initialization
    [h, w, nframes, nchans] = size(video);
    if isempty(warp_field) 
        display('* Initializing identity warping field');
        warp_field = zeros(h,w,nframes,3);
        video_warped = video; % No need to warp if the transformation is identity
    else
        video_warped = backward_warp(video, warp_field);
    end


    if isempty(video_regular) 
        display('* Initializing regular video');
        video_regular = video_warped;
    end

    res.warp_field = warp_field;

    for it_outer = 1:params.n_outer_iterations
        fprintf('* Outer iteration %2d of %2d\n', it_outer, params.n_outer_iterations);

        % Regularize video
        video_regular = update_regular_video(video_regular, video_warped, params);

        % warp video towards towards video_regular
        % warp_field = stwarp(video_warped, video_regular,params.stwarp);

        % video_warped = backward_warp(video_warped, warp_field);

    end
end % nlvv_ms_iteration
