function output = video_slice(video,slice_dim,slice_idx)
    assert(slice_dim < 4);
    assert(slice_dim > 0);

    if slice_dim == 1
        output = squeeze(video(slice_idx,:,:,:));
    elseif slice_dim == 2
        output = squeeze(video(:,slice_idx,:,:));
    else
        output = squeeze(video(:,:,slice_idx,:));
    end
end % video_slice
