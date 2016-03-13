function res = nlvv_ms_iteration(video, params, video_regular, warp_field)
% -------------------------------------------------------------------------
% -------------------------------------------------------------------------

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
    [video_regular, nnf] = update_regular_video(video_regular, video_warped, params);
    res.video_regular = video_regular;

    % warp video towards towards video_regular
    t = tic;
    fprintf('  - Compute warp\t\t');
    % TODO: give an initialization to the warping field
    warp_field = stwarp(video, video_regular,params.stwarp);
    res.warp_field    = warp_field;
    fprintf('%.1fs.\n', toc(t));

    % TODO: set warp field boundaries to 0 and remove inversions

    video_warped = backward_warp(video, warp_field);
    res.video_warped = video_warped;
end

end % nlvv_ms_iteration
