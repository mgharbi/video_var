% ----------------------------------------------------------------------------
% File:    exportRender.m
% Author:  Michael Gharbi <gharbi@mit.edu>
% Created: 2014-01-09
% ----------------------------------------------------------------------------
% 
% 
% 
% ---------------------------------------------------------------------------%

function exportRender(left,right,outputPath,renderPath)
    dims = size(left);
    nFr = size(right,3);
    nFl = size(left,3);
    if nFr > dims(3)
        dims(3) = nFr;
    end
    overlayPath     = numberedFilePath(outputPath, 0,'overlay-%04d.jpg',renderPath);
    sidePath        = numberedFilePath(outputPath, 0 ,'side-%04d.jpg',renderPath);

    % export image frames
    for frame = 1:dims(3)
        if frame<nFl
            f1 = squeeze(left(:,:,frame,:));
        else
            f1 = zeros(dims(1),dims(2),dims(4));
        end
        if frame<nFr
            f2 = squeeze(right(:,:,frame,:));
        else
            f2 = zeros(dims(1),dims(2),dims(4));
        end
        imSide = uint8(cat(2,f1,f2));
        imwrite(imSide,sprintf(sidePath,frame),'Quality',100);
        imOverlay = imfuse(f1(:,:,1),f2(:,:,1));
        imwrite(imOverlay,sprintf(overlayPath,frame),'Quality',100);
    end

    % ffmpeg rendering
    wOverlayOut = numberedFilePath(outputPath, 0,[renderPath '_overlay.mov']);
    sOut        = numberedFilePath(outputPath, 0,[renderPath '_sideCompare.mov']);

    sz = size(imSide); sz = sz(1:2);
    mkmov(sidePath,sOut,sz);
    sz = size(imOverlay); sz = sz(1:2);
    mkmov(overlayPath,wOverlayOut,sz);
end % exportRender
