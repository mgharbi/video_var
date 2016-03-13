function out = make_overlay(A,B)
    out = zeros(size(A),'uint8');
    for f = 1:size(A,3);
        oo = imfuse(im2double(squeeze(A(:,:,f,:))),im2double(squeeze(B(:,:,f,:))));
        out(:,:,f,:) = oo;
    end
end % make_overlay
