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
addpath('interpolate');

A = load_video('../data/derailleur_01.mov');
B = load_video('../data/derailleur_02.mov');


A = A(120:240,120:240,40:80,:);
B = B(120:240,120:240,40:80,:);

save_video(A,'../output/A.mp4');
save_video(B,'../output/B.mp4');


% Warp
warp = stwarp(A, B);

% Visualization
[spatial,temporal] = viz_warp(warp);
save_video(spatial,'../output/spatial.mp4');
save_video(temporal,'../output/temporal.mp4');

Bwarped = backward_interpolate(B,warp);
save_video(Bwarped,'../output/B_warped.mp4');

fused = make_overlay(A,B);
save_video(fused,'../output/overlay.mp4');

fused = make_overlay(A,Bwarped);
save_video(fused,'../output/overlay_warped.mp4');
