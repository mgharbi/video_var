function video = load_video(path)
    [basedir,file,ext] = fileparts(path);
    fromImages = false;

    if strcmp(ext,'.mat')
        video = load(path);
        video = video.video;
    else 
        if ~strcmp(ext,'')
            i_path = path;
            path = fullfile(basedir,[file '_tmp']);
            if ~exist(path,'dir')
                mkdir(path);
            end
            proto = fullfile(path,'%06d.png');
            cmd = sprintf('ffmpeg -i %s -y %s &>/dev/null', i_path, proto);
            unix(cmd);
        end
        files = dir(fullfile(path,'*.png'));
        nFrames = length(files);
        if nFrames == 0
            throw(MException('nlvv:VideoNotFound', 'no images found'));
        end
        names  = {files(:).name};
        im1 = imread(fullfile(path, names{1}));
        [h,w,c] = size(im1);
        video = zeros(h,w,nFrames,c,'single');
        for f = 1:nFrames
            frame = im2double(imread(fullfile(path, names{f})));
            video(:,:,f,:) = frame;
        end
        if ~strcmp(ext,'')
            rmdir(path, 's');
        end
    end
end % load_video
