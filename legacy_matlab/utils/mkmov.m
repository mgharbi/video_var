% ----------------------------------------------------------------------------
% File:    mkmov.m
% Author:  Michael Gharbi <gharbi@mit.edu>
% Created: 2014-01-09
% ----------------------------------------------------------------------------
% 
% Render a video from a sequence of files using ffmpeg .
% 
% ---------------------------------------------------------------------------%


function mkmov(inputPrototype,output,sz)
    if exist(output,'file')
        delete(output);
    end
    sz = round(sz/2)*2; %libx264 accpets only even size
    options = '-vcodec libx264 -profile:v high ';
    options = sprintf('%s -filter:v scale=%d:%d',options,sz(2), sz(1));
    redirect = '>/dev/null 2>&1';
    cmd = sprintf('ffmpeg -i %s  %s %s %s',inputPrototype,options,output,redirect);
    system(cmd);
end % function


