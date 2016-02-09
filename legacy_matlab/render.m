% ----------------------------------------------------------------------------
% File:    render.m
% Author:  Michael Gharbi <gharbi@mit.edu>
% Created: 2013-12-30
% ----------------------------------------------------------------------------
% 
% 
% 
% ---------------------------------------------------------------------------%

function render(videoA, videoB, warpField, exa,outputPath,renderPath,flowA)
    if nargin < 7
        flowA = [];
    end

    dims      = size(videoA);
    interpolate = false;
    medianFilter = true;

    [warpedB, outOfBounds] = backWarp3D(videoB,warpField,dims);

    fprintf('-- Amplifying difference using [%3d,%3d,%3d]\n',exa);

    % Perform amplification
    for i = 1:3
        warpField(:,:,:,i) = exa(i)*warpField(:,:,:,i);
    end


    if isempty(flowA)
        % Add cost term
        cost                   = single(videoA)-single(warpedB);

        cost                   = cost.*cost;
        cost                   = sum(cost,4);
        maxCost                = max(cost(:));
        cost(outOfBounds)     = 10*maxCost;
    else
        display('- using flow as cost map')
        cost = flowA(:,:,:,1).^2+flowA(:,:,:,2).^2;
        sig = 2;
        cost = exp(-cost/(2*sig*sig));
        cost = single(cost);
    end

    fprintf('-- Compute Warp...')
    [wField,mask] = forwardWarping(single(warpField), cost, [1 1 1]);
    fprintf('done.\n')
    if ~isempty(find(mask,1)) && interpolate 
        maskDilated                = imfilter(1-mask,ones(3,3,3));
        maskDilated(maskDilated>1) = 1;
        boundaries                 = maskDilated - 1 + mask;
        boundaries                 = logical(boundaries);
        mask                       = logical(mask);
        [X,Y,T] = meshgrid(1:dims(2), 1:dims(1), 1:dims(3));
        w       = wField(:,:,:,1);
        F       = scatteredInterpolant(X(boundaries),Y(boundaries),T(boundaries),double(w(boundaries)));
        for i = 1:3
            fprintf('\t - Interpolating channel %d...',i)
            w                    = wField(:,:,:,i);
            F.Values             = double(w(boundaries));
            interpd              = w;
            newData              = F(X(~mask),Y(~mask),T(~mask));
            interpd(~mask)       = newData;
            wField(:,:,:,i) = interpd;
            fprintf('done.\n')
        end
    end

    % if medianFilter
    %     fprintf('-- Median filtering\n')
    %     for k = 1:3
    %         wField(:,:,:,k) = medfilt3(wField(:,:,:,k),[5 5 3],'symmetric');
    %     end
    % end

    warped = backWarp3D(videoA,-wField);
    m = repmat(mask,[1,1,1,3]);
    warped(find(m==0))=0;


    fprintf('-- Output Warp...')
    exportRender(videoB,warped,outputPath,renderPath)
    fprintf('done.\n')
end
