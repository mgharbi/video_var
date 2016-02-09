% -------------------------------------------------------------------
% File:    run.m
% Author:  Michael Gharbi <gharbi@mit.edu>
% Created: 2016-02-02
% -------------------------------------------------------------------
% 
% 
% 
% ------------------------------------------------------------------#


addpath('../lib/mex');
addpath('viz_warp');
addpath('io');

A = load_video('../data/derailleur_01.mov');
B = load_video('../data/derailleur_02.mov');

% Warp
warp = stwarp(A, B);

% Visualization
[spatial,temporal] = viz_warp(warp);
save_video(spatial,'../output/spatial.mp4');
save_video(temporal,'../output/temporal.mp4');
