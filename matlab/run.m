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
params.knnf.patch_size_time = 21;
params.knnf.knn = 5; % default 20
params.n_outer_iterations  = 5;
params.n_inner_iterations  = 10; % default 10

params.stwarp.verbosity = 0;

% h = figure('Name', '3Dplot', 'Visible', 'off');

h = 71;
w = 71;
ph = 21;
pw = 21;
amp = 5;
nframes = 150;
freq1 = 0.05;
freq2 = 0.06;
rng(0);
video = oscillating_square([h,w],[ph,pw],linspace(freq1, freq2, nframes),amp);
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
t = tic;
res = nlvv(video,params);
t = toc(t);
res.time = t;

res.video = video;
res.params = params;
save(fullfile(globals.path.output, 'result.mat'), 'res');
