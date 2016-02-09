function field = loadWarpingField(path)
    f = fopen(path);
    dataType = fread(f,1,'int');
    h = fread(f,1,'int');
    w = fread(f,1,'int');
    nF = fread(f,1,'int');
    nC = fread(f,1,'int');
    field = zeros(h,w,nF,nC);
    if dataType == 1
        field(:) = fread(f, h*w*nF*nC,'single');
    else
        field(:) = fread(f, h*w*nF*nC,'double');
    end
    fprintf('-- loaded warping field %s.\n',path);
    fclose(f);
end % loadWarpingField
