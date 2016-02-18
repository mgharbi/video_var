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

% video = load_video('../data/rockettes01');
video = randn(20,30,10,3);

params.n_outer_iterations  = 1;
params.n_inner_iterations  = 1;
params.patch_size_spatial  = 15;
params.patch_size_temporal = 5;
params.nn_count            = 10;

res = nlvv(video,params);

