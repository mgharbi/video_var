function video = loadVideo(path)
    vr = VideoReader(path);
    sz = zeros(1,4);
    sz([1 2 4]) = size(read(vr,1));
    sz(3)       = vr.get('NumberOfFrames');

    video = zeros(sz);
    for i = 1:sz(3)
        video(:,:,i,:) = im2double(read(vr,i));
    end
    fprintf('-- loaded video %s.\n',path);
end % loadVideo
