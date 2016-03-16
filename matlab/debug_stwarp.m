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
rng(0);
video_regular = oscillating_square([h,w],[ph,pw],linspace(freq1, freq1, nframes),amp);

[h,w,nC,nF] = size(video);
x_coord = floor(w/2);

save_video(video, fullfile(debugdir, 'video.mp4'));
save_video(video_regular, fullfile(debugdir, 'video_regular.mp4'));

slice_yt = video_slice(video,2,x_coord);
imwrite(slice_yt, fullfile(debugdir, 'video_yt.png'));
slice_yt = video_slice(video_regular,2,x_coord);
imwrite(slice_yt, fullfile(debugdir, 'video_regular_yt.png'));


params = nlvv_params();
params.stwarp.verbosity = 2;
warp_field = stwarp(video_regular, video, params.stwarp);

[space,time] = viz_warp(warp_field);
slice_yt = video_slice(space,2,x_coord);
imwrite(slice_yt, fullfile(debugdir, 'space_yt.png'));
slice_yt = video_slice(time,2,x_coord);
imwrite(slice_yt, fullfile(debugdir, 'time_yt.png'));

video_warped = backward_warp(video, warp_field);

overlay = make_overlay(video,video_regular);
overlay_yt = video_slice(overlay,2,x_coord);
imwrite(overlay_yt, fullfile(debugdir,sprintf('overlay_yt.png')));
overlay = make_overlay(video_warped, video_regular);
overlay_yt = video_slice(overlay,2,x_coord);
imwrite(overlay_yt, fullfile(debugdir,sprintf('overlay_warped_yt.png')));

slice_yt = video_slice(video_warped,2,x_coord);
imwrite(slice_yt, fullfile(debugdir, 'video_warped_yt.png'));
save_video(video_warped, fullfile(debugdir, 'video_warped.mp4'));

video_exagerated = backward_warp(video, -2*warp_field);
slice_yt = video_slice(video_exagerated,2,x_coord);
imwrite(slice_yt, fullfile(debugdir, 'video_exagerated_yt.png'));
save_video(video_exagerated, fullfile(debugdir, 'video_exagerated.mp4'));

[warp_field2, j] = enforce_warp_consistency(warp_field);
j = normalize_video(j);
slice_yt = video_slice(j,2,x_coord);
imwrite(slice_yt, fullfile(debugdir, 'jacobian_yt.png'));
save_video(j, fullfile(debugdir, 'jacobian.mp4'));

[space,time] = viz_warp(warp_field2);
slice_yt = video_slice(space,2,x_coord);
imwrite(slice_yt, fullfile(debugdir, 'corrected_space_yt.png'));
slice_yt = video_slice(time,2,x_coord);
imwrite(slice_yt, fullfile(debugdir, 'corrected_time_yt.png'));

video_warped2 = backward_warp(video, warp_field2);
overlay = make_overlay(video_warped2, video_regular);
overlay_yt = video_slice(overlay,2,x_coord);
imwrite(overlay_yt, fullfile(debugdir,sprintf('overlay_warped_corrected_yt.png')));

slice_yt = video_slice(video_warped2,2,x_coord);
imwrite(slice_yt, fullfile(debugdir, 'video_warped_corrected_yt.png'));
save_video(video_warped2, fullfile(debugdir, 'video_warped_corrected.mp4'));
