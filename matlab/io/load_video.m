function video = load_video(path)
    % files = dir(fullfile(path,'*.png'));
    % nFrames = length(files);
    % names  = [files(:).name]
    % files(1)
    v = VideoReader(path);
    video = zeros(v.Height,v.Width,v.NumberOfFrames,3,'uint8');
    size(video);
    for i = 1:size(video,3)
        frame = read(v,i);
        video(:,:,i,:) = frame;
    end
end % load_video
