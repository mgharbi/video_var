function out = make_overlay(A,B)
    out = zeros(size(A),'uint8');
    max(A(:))
    for f = 1:size(A,3);
        oo = imfuse(squeeze(A(:,:,f,:)),squeeze(B(:,:,f,:)));
        out(:,:,f,:) = oo;
    end
end % make_overlay
