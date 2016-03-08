g = init();

v = load_video(fullfile(g.path.output,'nlvv_debug', '00_input.mp4'));
v2 = load_video(fullfile(g.path.output,'nlvv_debug', '34_3_regular.mp4'));
v3 = load_video(fullfile(g.path.output,'nlvv_debug', '30_2_warped.mp4'));

sl = video_slice(v,2,round(size(v,2)/2));
sl2 = video_slice(v2,2,round(size(v,2)/2));
sl3 = video_slice(v3,2,round(size(v,2)/2));
imwrite(sl, fullfile(g.path.output, 'input_slice.png'));
imwrite(sl2, fullfile(g.path.output, 'regular_slice.png'));
imwrite(sl3, fullfile(g.path.output, 'warped_slice.png'));
