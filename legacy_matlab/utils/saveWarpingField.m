function saveWarpingField(field, path)
    f = fopen(path, 'w');
    if isa(field, 'single')
        fprintf('float')
        fwrite(f,1,'int');
    else
        fprintf('double')
        fwrite(f,2,'int');
    end

    sz = size(field);
    fwrite(f,sz(1),'int');
    fwrite(f,sz(2),'int');
    fwrite(f,sz(3),'int');
    fwrite(f,sz(4),'int');
    if isa(field, 'single')
        fwrite(f,field,'single');
    else
        fwrite(f,field,'double');
    end
    fprintf('-- saved warping field %s.\n',path);
    fclose(f);
end % loadWarpingField
