% -------------------------------------------------------------------
% File:    run.m
% Author:  Michael Gharbi <gharbi@mit.edu>
% Created: 2016-02-16
% -------------------------------------------------------------------
% 
% 
% 
% ------------------------------------------------------------------#
close all;
clear all;

globals = init();

params = nlvv_params();

% params.knnf.knn = 2;
params.knnf.propagation_iterations = 3;
params.knnf.patch_size_space = 21;
params.knnf.patch_size_time = 27;
params.knnf.knn = 10; % default 20
params.n_outer_iterations  = 1;
params.n_inner_iterations  = 1; % default 10

% h = figure('Name', '3Dplot', 'Visible', 'off');

rng(0);
video = oscillating_square([51,21],[3,1],linspace(7, 12, 200),5);
mid = round(size(video,2)/2);

% params.plot3d = h;

% Reset debug dir
params.debug.output = fullfile(globals.path.output,'nlvv_debug');
if exist(params.debug.output,'dir')
    try
        rmdir(params.debug.output, 's');
    end
end

save_video(video, fullfile(params.debug.output, debug_path('input.mp4')), false);
sl = video_slice(video,2,round((size(video,2)+1)/2));
imwrite(sl, fullfile(params.debug.output, debug_path('yt_slice.png')));

size(video)
res = nlvv(video,params);

res.params = params;
save(fullfile(globals.path.output, 'result.mat'), 'res');
