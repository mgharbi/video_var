function [warp, jacobian] = enforce_warp_consistency(warp, n_iter, fsize)
    padsize = 10;
    warp = padarray(warp,[padsize,padsize,padsize,0],'replicate');

    wx = warp(:,:,:,1);
    wy = warp(:,:,:,2);
    wt = warp(:,:,:,3);

    jacobian = get_jacobian(wx, wy, wt, padsize);
    

    se = strel('disk',8);
    sigma = 2;
    n_iter = 30;
    for i = 1:n_iter
        % locate singular warping field
        singular_idx = imdilate(jacobian<0, se);

        smooth_wx = imgaussfilt3(wx, sigma);
        smooth_wy = imgaussfilt3(wy, sigma);
        smooth_wt = imgaussfilt3(wt, sigma);

        wx(singular_idx) = smooth_wx(singular_idx);
        wy(singular_idx) = smooth_wy(singular_idx);
        wt(singular_idx) = smooth_wt(singular_idx);

        jacobian = get_jacobian(wx, wy, wt, padsize);
    end

    if padsize > 0
        wx = wx(padsize+1:end-padsize,padsize+1:end-padsize,padsize+1:end-padsize);
        wy = wy(padsize+1:end-padsize,padsize+1:end-padsize,padsize+1:end-padsize);
        wt = wt(padsize+1:end-padsize,padsize+1:end-padsize,padsize+1:end-padsize);
    end

    warp = cat(4,wx,wy,wt);
end % enforce_warp_consistency

function jacobian = get_jacobian(wx,wy,wt,padsize)
    [hx,hy,ht] = get_derivative_filters3d();

    wx_dx = imfilter(wx,hx);
    wx_dy = imfilter(wx,hy);
    wx_dt = imfilter(wx,ht);

    wy_dx = imfilter(wy,hx);
    wy_dy = imfilter(wy,hy);
    wy_dt = imfilter(wy,ht);

    wt_dx = imfilter(wt,hx);
    wt_dy = imfilter(wt,hy);
    wt_dt = imfilter(wt,ht);

    jacobian = (1+wx_dx).*(1+wy_dy).*(1+wt_dt) ...
             + wy_dx.*wt_dy.*wx_dt ...
             + wt_dx.*wx_dy.*wy_dt ...
             - wt_dx.*(1+wy_dy).*wx_dt ...
             - wy_dx.*wx_dy.*(1+wt_dt) ...
             - (1+wx_dx).*wt_dy.*wy_dt;

    if padsize > 0
        jacobian(1:padsize,:,:)  = 0;
        jacobian(:, 1:padsize,:) = 0;
        jacobian(:,:, 1:padsize) = 0;

        jacobian(end-padsize+1:end,:,:)  = 0;
        jacobian(:,end-padsize+1:end,:)  = 0;
        jacobian(:,:,end-padsize+1:end)  = 0;
    end
end % get_jacobian
