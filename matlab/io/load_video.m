function video = load_video(path)
    v = VideoReader(path);
    video = zeros(v.Height,v.Width,v.NumberOfFrames,3,'uint8');
    size(video);
    read(v,1);
    for i = 1:size(video,3)
        frame = read(v,i);
        video(:,:,i,:) = frame;
        break
    end
end % load_video
