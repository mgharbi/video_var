function save_video(video,path, as_images)
    if nargin < 3
        as_images  = false;
    end
    [basename, finalfname, ext] = fileparts(path);

    tmpdir = fullfile(basename,finalfname);

    proto = fullfile(tmpdir,'%06d.png');
    if ~exist(tmpdir,'dir')
        mkdir(tmpdir)
    end

    for i = 1:size(video,3)
        fname = sprintf(proto, i);
        imwrite(squeeze(video(:,:,i,:)), fname);
    end

    if ~as_images
        cmd = sprintf('ffmpeg -i %s -y %s &>/dev/null', proto, path);
        unix(cmd);
        rmdir(tmpdir, 's');
    end
end % save_video
