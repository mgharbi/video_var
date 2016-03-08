function output = enforce_warp_consistency(warp, n_iter, fsize)
    warp = padarray(warp,[10,10,10,0],'replicate');

    wx = warp(:,:,:,1);
    wy = warp(:,:,:,2);
    wt = warp(:,:,:,3);

    hx = [-1 0 1; -2 0 2; -1 0 1];
    hy = hx';

end % enforce_warp_consistency
