globals = init();
video = oscillating_square();

dst  = fullfile(globals.path.data, 'synth');
if ~exist(dst,'dir')
    mkdir(dst)
end

save_video(video, fullfile(dst, 'square.mp4' ), false);
