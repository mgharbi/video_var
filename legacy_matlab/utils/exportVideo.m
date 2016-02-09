% ----------------------------------------------------------------------------
% File:    exportVideo.m
% Author:  Michael Gharbi <gharbi@mit.edu>
% Created: 2014-05-21
% ----------------------------------------------------------------------------
% 
% 
% 
% ---------------------------------------------------------------------------%

function exportVideo(video,path,name)
    dims = size(video);
    tempDir = fullfile(path,name);
    tempPath = fullfile(tempDir,sprintf('%s-%s.jpg',name,'%04d'));
    if ~exist(tempDir,'dir')
        mkdir(tempDir);
    end
    for frame = 1:dims(3)
        f = uint8(squeeze(video(:,:,frame,:)));
        imwrite(f,sprintf(tempPath,frame),'Quality',100);
    end
    outPath = fullfile(path,sprintf('%s.mov',name));
    mkmov(tempPath,outPath,dims(1:2));
    rmdir(tempDir,'s');
end % exportVideo
