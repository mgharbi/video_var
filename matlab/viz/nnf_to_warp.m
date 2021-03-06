function warp = nnf_to_warp(nnf, knnf_params)
% Convert a NNF (with absolute integer indexing) to a float, relative warp
% field.

    [h,w,nF,nC] = size(nnf);
    assert(mod(nC,3) == 0);

    n_nnf = nC/3;

    [x,y,t] = meshgrid(1:w,1:h,1:nF);
    vals = cat(4,x,y,t);
    vals = repmat(vals,[1,1,1,n_nnf]);

    warp = single(nnf)-vals;

    warp(end-knnf_params.patch_size_space+1:end,:,:,:)  = 0;
    warp(:, end-knnf_params.patch_size_space+1:end,:,:) = 0;
    warp(:,:, end-knnf_params.patch_size_time+1:end,:)  = 0;

end % nnf_to_warp
