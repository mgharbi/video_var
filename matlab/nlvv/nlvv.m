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

video_current = video;
video_regular = [];
warp_field    = [];

for lvl = 1:n_pyr_levels
    % rescale image to current level

    res = nlvv_ms_iteration(video_current,params,
            video_regular, warp_field);

    % [img_regular, img_warped, ux, uy] = NonLocalVarMain(imgCur, param, img_regular, ux, uy);
    % Res(i-Smin+1).ux = (1/sf)^(i) * ResizeConsistentOut(ux, sf^(i), 1, [Q,R]);
    % Res(i-Smin+1).uy = (1/sf)^(i) * ResizeConsistentOut(uy, sf^(i), 1, [Q,R]);
end

end % non_local_video_var

