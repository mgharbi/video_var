function warp_field = reset_warp_boundaries(warp_field)
    warp_field(1,:,:,:)   = 0;
    warp_field(end,:,:,:) = 0;
    warp_field(:,1,:,:)   = 0;
    warp_field(:,end,:,:) = 0;
    warp_field(:,:,1,:)   = 0;
    warp_field(:,:,end,:) = 0;
end % reset_warp_boundaries
