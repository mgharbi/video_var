% ----------------------------------------------------------------------------
% File:    numberedFilePath.m
% Author:  Michael Gharbi <gharbi@mit.edu>
% Created: 2013-12-30
% ----------------------------------------------------------------------------
% 
% Auxialiary function that generates path for output data.
% 
% ---------------------------------------------------------------------------%


function path = numberedFilePath(outputPath,index,fileNameModel,folderName)
    if nargin < 3
        fileNameModel = 'temp';
    end
    if nargin < 4
        folderName = '';
    end

    outDir = fullfile(outputPath,folderName);

    % make sure the save path is valid
    if ~exist(outDir,'dir')
        mkdir(outDir);
    end

    if index > 0
        path = fullfile(outDir,sprintf(fileNameModel,index));
    else
        path = fullfile(outDir,fileNameModel);
    end
end % numberedFilePath
