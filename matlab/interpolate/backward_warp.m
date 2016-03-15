function output = backward_warp(B,w)
    width = size(w,2);
    height = size(w,1);
    nF = size(w,3);
    [X,Y,Z] = meshgrid(1:width, 1:height, 1:nF);

    % Coordinates to sample from
    Xb = X + w(:,:,:,1);
    Yb = Y + w(:,:,:,2);
    Zb = Z + w(:,:,:,3);

    % Clamp to image boundaries
    % Xb(Xb<1) = 1;
    % Yb(Yb<1) = 1;
    % Zb(Zb<1) = 1;
    % Xb(Xb>width) = width;
    % Yb(Yb>height) = height;
    % Zb(Zb>nF) = nF;

    out1 = interp3(X,Y,Z,B(:,:,:,1),Xb,Yb,Zb);
    out2 = interp3(X,Y,Z,B(:,:,:,2),Xb,Yb,Zb);
    out3 = interp3(X,Y,Z,B(:,:,:,3),Xb,Yb,Zb);

    output = cat(4,out1,out2,out3);

end % backward_interpolate
