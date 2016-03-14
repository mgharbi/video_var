close all;
clear all;

globals = init();

res = load(fullfile(globals.path.output, 'result.mat'));
res = res.res;

debugdir = fullfile(globals.path.output,'debug');
if exist(debugdir,'dir')
    try
        rmdir(debugdir, 's');
    end
end
mkdir(debugdir)

[h,w,nC,nF] = size(res.video);
x_coord = floor(w/2);

% save_video(res.video, fullfile(debugdir, 'video'),true);
% save_video(res.video_regular, fullfile(debugdir, 'video_regular'),true);
% save_video(res.video_warped, fullfile(debugdir, 'video_warped'),true);

[space,time] = viz_warp(res.warp_field);

% save_video(space, fullfile(debugdir, 'space'),true);
% save_video(time, fullfile(debugdir, 'time'),true);

slice_yt = video_slice(space,2,x_coord);
imwrite(slice_yt, fullfile(debugdir, 'spacewarp_yt.png'));
slice_yt = video_slice(time,2,x_coord);
imwrite(slice_yt, fullfile(debugdir, 'timewarp_yt.png'));

slice_yt = video_slice(res.video,2,x_coord);
imwrite(slice_yt, fullfile(debugdir, 'input_yt.png'));
slice_yt = video_slice(res.video_regular,2,x_coord);
imwrite(slice_yt, fullfile(debugdir, 'regular_yt.png'));
slice_yt = video_slice(res.video_warped,2,x_coord);
imwrite(slice_yt, fullfile(debugdir, 'warped_yt.png'));


overlay = make_overlay(res.video,res.video_regular);
overlay_yt = video_slice(overlay,2,x_coord);
imwrite(overlay_yt, fullfile(debugdir,sprintf('overlay_yt.png')));
overlay = make_overlay(res.video_warped,res.video_regular);
overlay_yt = video_slice(overlay,2,x_coord);
imwrite(overlay_yt, fullfile(debugdir,sprintf('overlay_warped_yt.png')));

exagerated = backward_warp(res.video, -2*res.warp_field);
slice_yt = video_slice(exagerated,2,x_coord);
imwrite(slice_yt, fullfile(debugdir, 'exagerated_yt.png'));

w = normalize_video(res.w);
for i = 1:size(res.w,4)
    slice_yt = video_slice(res.w(:,:,:,i),2,x_coord);
    slice_yt = normalize_video(slice_yt);
    imwrite(slice_yt, fullfile(debugdir, sprintf('weights_%d_yt.png', i)));
    save_video(w(:,:,:,1), fullfile(debugdir, sprintf('weights_%d', i)),true);
end

nnfwarp = nnf_to_warp(res.nnf,res.params.knnf);
for i = 1:size(nnfwarp,4)/3
    [space,time] = viz_warp(nnfwarp(:,:,:,3*(i-1)+1:3*i));
    slice_yt = video_slice(space,2,x_coord);
    imwrite(slice_yt, fullfile(debugdir,sprintf('nnf_%d_space_yt.png', i)));
    slice_yt = video_slice(time,2,x_coord);
    imwrite(slice_yt, fullfile(debugdir,sprintf('nnf_%d_time_yt.png', i)));
end
