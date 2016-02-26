function res = nlvv_ms_iteration(video,params,video_regular,warp_field)
    % Initialization
    [h, w, nframes, nchans] = size(video);
    if isempty(warp_field) 
        display('Initializing identity warping field');
        warp_field = zeros(h,w,nframes,3);
    end

    video_warped = backward_warp(video, warp_field);

    if isempty(video_regular) 
        display('Initializing regular video');
        video_regular = video_warped;
    end

    res.warp_field = warp_field;

    for it_outer = 1:params.n_outer_iterations
        video_regular = update_regular_video(video_regular, video_warped, params);
    end
end % nlvv_ms_iteration
