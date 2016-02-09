function [warped, mask] = backWarp3D(data, warper,sz)
    if nargin<3
        sz = size(data);
        if size(sz) < 4
            sz(3) = 1;
        end
    end
    sz(4) = size(data,4);
    [X,Y,T] = meshgrid(1:sz(2), 1:sz(1), 1:sz(3));
    X = X+warper(:,:,:,1);
    Y = Y+warper(:,:,:,2);
    T = T+warper(:,:,:,3);

    mask = (X<1) | (X>sz(2)) | (Y<1) | (Y>sz(1)) | (T<1) | (T>sz(3));

    warped = zeros(sz);

    data = double(data);

    for i = 1:sz(4)
        warped(:,:,:,i) = interp3(data(:,:,:,i),X,Y,T);
    end

    warped(isnan(warped)) = 0;
end % backWarp
