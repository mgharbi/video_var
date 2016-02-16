init();

B = load_video('../data/rockettes01', true);
[h,w,nF,nC] = size(B);

[x,y,t] = meshgrid(1:w, 1:h, 1:nF);

wx = sin(3*2*pi*(y-h/2)/h).*(t/nF);

wx = -wx*10;

% plot(w(1,:,1))
% plot(w(1,:,floor(nF/2)))
% hold on 
% plot(w(1,:,nF))

warp = zeros(size(B));
warp(:,:,:,1) = wx;

Bwarped = backward_interpolate(B,warp);
save_video(Bwarped,'../output/rockettes_warped.mp4');

[spatial,temporal] = viz_warp(warp);
save_video(spatial,'../output/rockettes_spatial_gt.mp4');
save_video(temporal,'../output/rockettes_temporal_gt.mp4');
