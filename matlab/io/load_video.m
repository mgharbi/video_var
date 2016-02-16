function video = load_video(path, fromImages)
    if nargin < 2
        fromImages = false;
    end

    if fromImages
        files = dir(fullfile(path,'*.png'));
        nFrames = length(files);
        if nFrames == 0
            throw 'no images found'
        end
        names  = {files(:).name};
        im1 = imread(fullfile(path, names{1}));
        [h,w,c] = size(im1);
        video = zeros(h,w,nFrames,c,'uint8');
        for f = 1:nFrames
            frame = imread(fullfile(path, names{f}));
            video(:,:,f,:) = frame;
        end
    else
        v = VideoReader(path);
        video = zeros(v.Height,v.Width,v.NumberOfFrames,3,'uint8');
        size(video);
        for i = 1:size(video,3)
            frame = read(v,i);
            video(:,:,i,:) = frame;
        end
    end
end % load_video
