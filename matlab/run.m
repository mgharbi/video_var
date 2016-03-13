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
params.knnf.patch_size_space = 11;
params.knnf.patch_size_time = 17;
params.knnf.knn = 3; % default 20
params.n_outer_iterations  = 1;
params.n_inner_iterations  = 1; % default 10

params.stwarp.verbosity = 1;

% h = figure('Name', '3Dplot', 'Visible', 'off');

rng(0);
nframes = 80;
freq1 = 3;
freq2 = 5;
video = oscillating_square([51,21],[3,1],linspace(freq1, freq2, nframes),5);
mid = round(size(video,2)/2);

% params.plot3d = h;

% Reset debug dir
params.debug.output = fullfile(globals.path.output,'nlvv_debug');
% if exist(params.debug.output,'dir')
%     try
%         rmdir(params.debug.output, 's');
%     end
% end

save_video(video, fullfile(params.debug.output, debug_path('input.mp4')), false);
sl = video_slice(video,2,round((size(video,2)+1)/2));
imwrite(sl, fullfile(params.debug.output, debug_path('yt_slice.png')));

size(video)
res = nlvv(video,params);

res.video = video;
res.params = params;
save(fullfile(globals.path.output, 'result.mat'), 'res');
