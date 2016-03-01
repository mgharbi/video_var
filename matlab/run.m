% -------------------------------------------------------------------
% File:    run.m
% Author:  Michael Gharbi <gharbi@mit.edu>
% Created: 2016-02-16
% -------------------------------------------------------------------
% 
% 
% 
% ------------------------------------------------------------------#

globals = init();

rng(0);
video = oscillating_square(51,11,1:3);

params = nlvv_params();

% Reset debug dir
params.debug.output = fullfile(globals.path.output,'nlvv_debug');
if exist(params.debug.output,'dir')
    rmdir(params.debug.output, 's');
end

save_video(video, fullfile(params.debug.output, debug_path('input.mp4')), false);

res = nlvv(video,params);
