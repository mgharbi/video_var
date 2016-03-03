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
params.knnf.patch_size_space = 3;
params.knnf.patch_size_time = 5;
params.knnf.knn = 10; % default 20
params.n_outer_iterations  = 2; % default 10
params.n_inner_iterations  = 5;

% h = figure('Name', '3Dplot', 'Visible', 'off');

rng(0);
video = oscillating_square(51,11,linspace(2, 3,60));
mid = round(size(video,2)/2);
video = video(:,mid-5:mid+5,:,:);

% params.plot3d = h;

% Reset debug dir
params.debug.output = fullfile(globals.path.output,'nlvv_debug');
if exist(params.debug.output,'dir')
    rmdir(params.debug.output, 's');
end

save_video(video, fullfile(params.debug.output, debug_path('input.mp4')), false);
sl = video_slice(video,2,round(size(video,2)/2));
imwrite(sl, fullfile(params.debug.output, debug_path('yt_slice.png')));
sl = video_slice(video,1,round(size(video,1)/2));
imwrite(sl, fullfile(params.debug.output, debug_path('xt_slice.png')));

size(video)
res = nlvv(video,params);
% space_time_scatter(h, [20,20,20; 25,26,30], [1,.3], [1, .2, .2]);
% h.Visible = 'on';
