addpath('../lib/mex');
addpath('viz_warp');
addpath('io');

A = load_video('../data/derailleur_01.mov');
B = load_video('../data/derailleur_02.mov');

A = A(1:30,1:30,1:30,:);
B = B(1:30,1:30,1:30,:);
warp = stwarp(A, B);
size(warp)

[spatial,temporal] = viz_warp(warp);

save_video(spatial,'../output/spatial.mp4');
save_video(temporal,'../output/temporal.mp4');
