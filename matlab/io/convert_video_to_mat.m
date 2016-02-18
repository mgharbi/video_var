function output = convert_video_to_mat(path)
    [basedir,file,ext] = fileparts(path);
    video = load_video(path);
    save(fullfile(basedir,[file '.mat']),'video');
end % convert_video_to_mat
