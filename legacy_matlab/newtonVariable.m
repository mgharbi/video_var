addpath(genpath(pwd));

filename      = 'newtonVariable';
dateStr       = '2014-05-14';
outputPath    = sprintf('../output/%s/%s',dateStr,filename);
warpFieldPath = sprintf('../output/%s/%s/%s.stw', dateStr, filename, filename);
nnfPath       = sprintf('../output/%s/%s/%s_nnf.stw', dateStr, filename, filename);
videoApath    = sprintf('../data/%s_01.mov', filename);
videoBpath    = sprintf('../data/%s_02.mov', filename);
outPath       = sprintf('../output/%s/%s/%s-matlab.stw', dateStr, filename, filename);
transPath     = sprintf('../output/%s/%s/%s_trans.stw', dateStr, filename, filename);
xformPath     = sprintf('../output/%s/%s/%s_xform.stw', dateStr, filename, filename);

videoA = uint8(255*loadVideo(videoApath));
videoB = uint8(255*loadVideo(videoBpath));

sa = size(videoA);
sb = size(videoB);

uvw = loadWarpingField(warpFieldPath);
render(videoA,videoB,uvw, [-2 -2 1], outputPath,'space2');
render(videoA,videoB,uvw, [1 1 1], outputPath,'align');
return;
uvw = zeros(sa);

imshow(imfuse(squeeze(videoA(:,:,59,:)),squeeze(videoB(:,:,28,:))));

times01 = [1 59 106];
times02 = [1 28 49];
interpolatedW = interp1(times01,times02,1:sa(3),'linear');


uTimes = [1 59 75 106];
u = [0 9 0 0];
interpolatedU = interp1(uTimes,u,1:sa(3),'linear');

for i=1:sa(3)
    uvw(:,:,i,3) = interpolatedW(i)-i;
    % uvw(:,:,i,1) = interpolatedU(i);
end

warpedB = uint8(backWarp3D(videoB,uvw,sa));
% saveWarpingField(single(uvw), transPath);
%


% csh_wid = 16;
% [uv1,flag1] = CSH_nn_flow_base(I0,I1,[],csh_wid,1);
% [histmat_csh, top_uv2] = Dominant_Offset(uv1,flag1);
% top_homo = Dominant_Transform(uv1);
% [uv,uv1,uv2] = knn_flow(I0,I1,top_uv2,top_homo);
% uvo = estimate_flow_interface2(I0,I1, 'classic+nl-fast', [], uv);
% flow(:,:,i,1) = uvo(:,:,1);
% flow(:,:,i,2) = uvo(:,:,2);

for frame = 1:sa(3)
% frame = 10;
    I0 = squeeze(videoA(:,:,frame,:));
    I1 = squeeze(warpedB(:,:,frame,:));
    uvo = estimate_flow_interface(I0,I1, 'ba-brightness', []);
    uvw(:,:,frame,1) = uvo(:,:,1);
    uvw(:,:,frame,2) = uvo(:,:,2);
end

saveWarpingField(single(uvw), warpFieldPath);

% I1w = uint8(backWarp2D(double(I1),uvo));
% imshow(imfuse(I0,I1));
% figure;
% imshow(imfuse(I0,I1w));
% figure;
% imshow(I1w)
% figure;
% imshow(flowToColor(uvo))
%
