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

finest_res = [h,w,nF];

scales = params.scale_max:-1:params.scale_min;
ref_scale = [1.0, 1.0, 1.0];
for lvl = scales
    disp(['Level:' num2str(lvl) ' min level: ' num2str(params.scale_min)]);

    dst_scale = ref_scale*(params.pyramid_ratio)^lvl;
    video_current = consistent_video_resize(video, ref_scale, dst_scale, finest_res);

    % Upsample the previous regular video and warping_field
    % (if not at the deepest level)
    if lvl < params.scale_max % not at the coarset res, upsample
        src_scale = ref_scale*(params.pyramid_ratio)^(lvl+1);
        dst_scale = ref_scale*(params.pyramid_ratio)^lvl;
        video_regular = consistent_video_resize(video_regular, src_scale, dst_scale, finest_res); 
        warp_field = consistent_video_resize(warp_field, src_scale, dst_scale, finest_res); 
        warp_field = warp_field*1/params.pyramid_ratio
    end

    % Process this scale
    res = nlvv_ms_iteration(video_current,params,...
            video_regular, warp_field);

    % TODO: save_upsized_maps
end

end % non_local_video_var

