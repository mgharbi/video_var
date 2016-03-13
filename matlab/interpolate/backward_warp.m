function output = backward_warp(B,w)
    [X,Y,Z] = meshgrid(1:size(w,2), 1:size(w,1), 1:size(w,3));

    % Coordinates to sample from
    Xb = X + w(:,:,:,1);
    Yb = Y + w(:,:,:,2);
    Zb = Z + w(:,:,:,3);

    out1 = interp3(X,Y,Z,B(:,:,:,1),Xb,Yb,Zb);
    out2 = interp3(X,Y,Z,B(:,:,:,2),Xb,Yb,Zb);
    out3 = interp3(X,Y,Z,B(:,:,:,3),Xb,Yb,Zb);

    output = cat(4,out1,out2,out3);

end % backward_interpolate
