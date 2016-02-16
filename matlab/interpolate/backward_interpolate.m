function output = backward_interpolate(B,w)
    [X,Y,Z] = meshgrid(1:size(w,2), 1:size(w,1), 1:size(w,3));

    Xb = X + w(:,:,:,1);
    Yb = Y + w(:,:,:,2);
    Zb = Z + w(:,:,:,3);


    out1 = interp3(X,Y,Z,single(B(:,:,:,1)),Xb,Yb,Zb);
    out2 = interp3(X,Y,Z,single(B(:,:,:,2)),Xb,Yb,Zb);
    out3 = interp3(X,Y,Z,single(B(:,:,:,3)),Xb,Yb,Zb);

    output = uint8(cat(4,out1,out2,out3));

end % backward_interpolate
