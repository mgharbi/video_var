% -------------------------------------------------------------------
% File:    run.m
% Author:  Michael Gharbi <gharbi@mit.edu>
% Created: 2016-02-02
% -------------------------------------------------------------------
% 
% 
% 
% ------------------------------------------------------------------#


globals = init();

A = load_video('../data/rockettes01');
B = load_video('../data/rockettes02');

params = stwarp_params();

% Warp
warp = stwarp(A, B,params);

% Visualization
[spatial,temporal] = viz_warp(warp);
save_video(spatial,'../output/spatial.mp4');
save_video(temporal,'../output/temporal.mp4');

Bwarped = backward_warp(B,warp);
save_video(Bwarped,'../output/B_warped.mp4');

fused = make_overlay(A,B);
save_video(fused,'../output/overlay.mp4');

fused = make_overlay(A,Bwarped);
save_video(fused,'../output/overlay_warped.mp4');
