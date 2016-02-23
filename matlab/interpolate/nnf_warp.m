function output = backward_warp(B,sizeA, w)
    if sizeA(3) == 1
        [X,Y] = meshgrid(1:sizeA(2), 1:sizeA(1));

        % Coordinates to sample from
        Xb = single(w(:,:,:,1)+1);
        Yb = single(w(:,:,:,2)+1);


        out1 = interp2(X,Y,single(B(:,:,:,1)),Xb,Yb);
        out2 = interp2(X,Y,single(B(:,:,:,2)),Xb,Yb);
        out3 = interp2(X,Y,single(B(:,:,:,3)),Xb,Yb);

        output = uint8(cat(3,out1,out2,out3));
    else
        [X,Y,Z] = meshgrid(1:sizeA(2), 1:sizeA(1), 1:sizeA(3));

        % Coordinates to sample from
        Xb = single(w(:,:,:,1)+1);
        Yb = single(w(:,:,:,2)+1);
        Zb = single(w(:,:,:,3)+1);


        out1 = interp3(X,Y,Z,single(B(:,:,:,1)),Xb,Yb,Zb);
        out2 = interp3(X,Y,Z,single(B(:,:,:,2)),Xb,Yb,Zb);
        out3 = interp3(X,Y,Z,single(B(:,:,:,3)),Xb,Yb,Zb);

        output = uint8(cat(4,out1,out2,out3));

    end
end % backward_interpolate
