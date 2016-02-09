% ----------------------------------------------------------------------------
% File:    dominantTranslations.m
% Author:  Michael Gharbi <gharbi@mit.edu>
% Created: 2014-03-19
% ----------------------------------------------------------------------------
% 
% 
% 
% ---------------------------------------------------------------------------%


function [h, top_uvw] = dominantTranslations2D(nnf,valid_flag, frame)
    K = 40; % number of top matches to keep
    % try 200

    u = nnf(:,:,frame,1);
    v = nnf(:,:,frame,2);
    w = nnf(:,:,frame,3);
    valid_flag = valid_flag(:,:,frame);

    if exist('valid_flag','var') && ~isempty(valid_flag)
        ind = find(valid_flag==1);
        u=u(ind); v=v(ind); w = w(ind);
    end

    maxUV   = max( max(abs(u)), max(abs(v)) )+1;
    maxW    = max(abs(w))+1;
    space_i = linspace(-maxUV,maxUV,2*maxUV+1);
    time_i = linspace(-maxW,maxW,2*maxW+1);

    h = ndHistc(cat(2,u,v,w),space_i,space_i,time_i);

    [U,V,W] = meshgrid(space_i(1:end-1),space_i(1:end-1), time_i(1:end-1));

    top_uvw = dominant_uvw(h,U,V,W,K);
end % dominantTranslations

function uvw = dominant_uvw(h, U, V, W, K)
if ~exist('K','var')
    K = 15;
end
    h = h(:);
    idx = find(h~=0);
    K = min(K,length(idx));

    [b,ix] = sort(h,'descend');

    ix = ix(1:K);
    b  = b(1:K);

    uvw = cat(2,U(ix),V(ix),W(ix),b);
end % dominant_uvw
