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
    [video_regular, nnf, w, inner_res] = update_regular_video(video_regular, video_warped, params);
    res.video_regular = video_regular;
    res.nnf = nnf;
    res.w = w;

    res.iter{it_outer} = inner_res;

    % warp video towards towards video_regular
    t = tic;
    fprintf('  - Compute warp\t\t');
    % TODO: give an initialization to the warping field
    warp_field = stwarp(video, video_regular, params.stwarp, warp_field);
    % warp_field = stwarp(video_regular, video, params.stwarp, warp_field);
    res.warp_field    = warp_field;
    fprintf('%.1fs.\n', toc(t));
    fprintf('    range: %.3f,%.3f\n', min(warp_field(:)), max(warp_field(:)) );

    % TODO: set warp field boundaries to 0 and remove inversions

    video_warped = backward_warp(video, warp_field);
    res.video_warped = video_warped;
end

end % nlvv_ms_iteration
