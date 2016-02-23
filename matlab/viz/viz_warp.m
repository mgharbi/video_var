function [spatial, temporal] = viz_warp(warp)
    spatial = flowToColor(warp(:,:,:,1:2));
    temporal = flowToColor(warp(:,:,:,3:3));
end % viz_warp
