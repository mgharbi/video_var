close all;
clear all;

globals = init();

debugdir = fullfile(globals.path.output,'debug');
if exist(debugdir,'dir')
    try
        rmdir(debugdir, 's');
    end
end
mkdir(debugdir)

nframes = 80;
freq1 = 4;
freq2 = 4.3
rng(0);
video = oscillating_square([51,21],[3,1],linspace(freq1, freq2, nframes),5);
rng(0);
video_regular = oscillating_square([51,21],[3,1],linspace(freq1, freq1, nframes),5);

% video(:,:,1:10,:) = 0;
% video_regular(:,:,1:10,:) = 0;
% video(:,:,end-10:end,:) = 0;
% video_regular(:,:,end-10:end,:) = 0;

[h,w,nC,nF] = size(video);
x_coord = floor(w/2);

save_video(video, fullfile(debugdir, 'video.mp4'));
save_video(video_regular, fullfile(debugdir, 'video_regular.mp4'));

slice_yt = video_slice(video,2,x_coord);
imwrite(slice_yt, fullfile(debugdir, 'video_yt.png'));
slice_yt = video_slice(video_regular,2,x_coord);
imwrite(slice_yt, fullfile(debugdir, 'video_regular_yt.png'));

params = nlvv_params();
params.stwarp.verbosity = 1;
warp_field = stwarp(video, video_regular, params.stwarp);

[space,time] = viz_warp(warp_field);
% save_video(space, fullfile(debugdir, 'space'),true);
% save_video(time, fullfile(debugdir, 'time'),true);

slice_yt = video_slice(space,2,x_coord);
imwrite(slice_yt, fullfile(debugdir, 'space_yt.png'));
slice_yt = video_slice(time,2,x_coord);
imwrite(slice_yt, fullfile(debugdir, 'time_yt.png'));

video_warped = backward_warp(video_regular, warp_field);

overlay = make_overlay(video,video_regular);
overlay_yt = video_slice(overlay,2,x_coord);
imwrite(overlay_yt, fullfile(debugdir,sprintf('overlay_yt.png')));
overlay = make_overlay(video, video_warped);
overlay_yt = video_slice(overlay,2,x_coord);
imwrite(overlay_yt, fullfile(debugdir,sprintf('overlay_warped_yt.png')));

slice_yt = video_slice(video_warped,2,x_coord);
imwrite(slice_yt, fullfile(debugdir, 'video_warped_yt.png'));
