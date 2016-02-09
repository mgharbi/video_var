function [warped, mask] = backWarp2D(data, warper)
    sz = size(data);
    if size(sz) < 3
        sz(3) = 1;
    end
    [X,Y] = meshgrid(1:sz(2), 1:sz(1));
    X = X+warper(:,:,1);
    Y = Y+warper(:,:,2);

    mask = (X<0) | (X>sz(2)) | (Y<0) | (Y>sz(1));

    warped = zeros(sz);

    for i = 1:sz(3)
        warped(:,:,i) = interp2(data(:,:,i),X,Y);
    end

    warped(isnan(warped)) = 0;
end % backWarp
