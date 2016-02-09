% ----------------------------------------------------------------------------
% File:    loadSLIC.m
% Author:  Michael Gharbi <gharbi@mit.edu>
% Created: 2014-05-22
% ----------------------------------------------------------------------------
% 
% 
% 
% ---------------------------------------------------------------------------%

function slic = loadSLIC(sz,prototype)
    slic = zeros(sz(1),sz(2),sz(3));
    for i = 1:sz(3)
        %% TODO: figure out why thus hack is needed, Cxx and Matlab versions
        %dont read the same number of frames?
        try
            s = loadSLICimg(sz(1:2), sprintf(prototype,i));
            s = reshape(s,[sz(2) sz(1)])';
            s = reshape(s,[sz(1) sz(2)]);
            slic(:,:,i) = s;
        catch 
            if i==sz(3)
            slic(:,:,i) = slic(:,:,i-1);
            end
        end
    end
    fprintf('-- loaded slic.\n');
end % loadSLIC

function slic = loadSLICimg(sz,path)
    f = fopen(path);
    slic = zeros(sz(1),sz(2));
        slic(:) = fread(f, numel(slic),'int');
    fclose(f);
end % loadWarpingField
