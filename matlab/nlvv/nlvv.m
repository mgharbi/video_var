function res = nlvv(video,params)
% -------------------------------------------------------------------
% File:    nlvv.m
% Author:  Michael Gharbi <gharbi@mit.edu>
% Created: 2016-02-18
% -------------------------------------------------------------------
% 
% Non-local video variation, entry point to the algorithm
% 
% -------------------------------------------------------------------


% - Input check and params setting ----------------------------------

n_pyr_levels = 1;

[h, w, nframes, nchans] = size(video);

video_current = video;
video_regular = [];
warp_field    = [];

scales = params.scale_max:-1:params.scale_min;
for lvl = scales
    disp(['Level:' num2str(lvl) ' min level: ' num2str(params.scale_min)]);

    % rescale video to current level
    video_current = consistent_video_resize(video);

    % Upsample the previous regular video and warping_field
    % (if not at the deepest level)
    if lvl < params.scale_max
        consistent_video_resize()
    end

    res = nlvv_ms_iteration(video_current,params,...
            video_regular, warp_field);
    % save_upsized_maps

    % [img_regular, img_warped, ux, uy] = NonLocalVarMain(imgCur, param, img_regular, ux, uy);
    % Res(i-Smin+1).ux = (1/sf)^(i) * ResizeConsistentOut(ux, sf^(i), 1, [Q,R]);
    % Res(i-Smin+1).uy = (1/sf)^(i) * ResizeConsistentOut(uy, sf^(i), 1, [Q,R]);
end

end % non_local_video_var

