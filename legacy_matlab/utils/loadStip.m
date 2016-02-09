function stip = loadStip(path)
    f = fopen(path);
    n = 0;
    while ~feof(f)
        fgetl(f);
        n = n+1;
    end
    n = n - 3;
    fclose(f);

    stip = zeros(n,165);
    
    f = fopen(path);
    for i=1:3
        fgetl(f);
    end
    i = 1;
    while ~feof(f)
        tline = str2num(fgetl(f));
        stip(i,1:3) = tline(2:4); % x,y,t
        stip(i,4:end) = tline(8:end); % 72hog, 90 hof
        i = i+1;
    end
    fclose(f);
end % loadWarpingField
