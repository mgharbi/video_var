% ----------------------------------------------------------------------------
% File:    computeOpticalFlow.m
% Author:  Michael Gharbi <gharbi@mit.edu>
% Created: 2014-05-14
% ----------------------------------------------------------------------------
% 
% 
% 
% ---------------------------------------------------------------------------%

function flow = computeOpticalFlow(video)
    [h,w,nF,~] = size(video);
    flow = zeros(h,w,nF,2);
    csh_wid = 16;
    for i = 1:nF-1
        I0 = squeeze(video(:,:,i,:));
        I1 = squeeze(video(:,:,i+1,:));
        [uv1,flag1] = CSH_nn_flow_base(I0,I1,[],csh_wid,1);
        [~, top_uv2] = Dominant_Offset(uv1,flag1);
        top_homo = Dominant_Transform(uv1);
        [uv,~,~] = knn_flow(I0,I1,top_uv2,top_homo);
        uvo = estimate_flow_interface2(I0,I1, 'classic+nl-fast', [], uv);
        flow(:,:,i,1) = uvo(:,:,1);
        flow(:,:,i,2) = uvo(:,:,2);
        fprintf('- computing NNF-based flow - %03d/%03d\n', i, nF-1)
    end
    flow = single(flow);
end % computeOpticalFlow
