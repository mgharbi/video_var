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

params = nlvv_params();

res = nlvv(video,params);
