function save_video(video,path)
    [basename, finalfname, ext] = fileparts(path);
    tmpdir = fullfile(basename,'tmp_video');
    proto = fullfile(tmpdir,'%06d.png');
    if ~exist(tmpdir,'dir')
        mkdir(tmpdir)
    end
    for i = 1:size(video,3)
        fname = sprintf(proto, i);
        imwrite(squeeze(video(:,:,i,:)), fname);
    end
    cmd = sprintf('ffmpeg -i %s -y %s &>/dev/null', proto, path);
    unix(cmd);
    rmdir(tmpdir, 's');
end % save_video
