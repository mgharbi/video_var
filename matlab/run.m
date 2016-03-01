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

video = oscillating_square(51,11,1);

params = nlvv_params();

res = nlvv(video,params);
