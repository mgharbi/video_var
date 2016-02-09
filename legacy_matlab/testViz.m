
addpath(genpath(pwd));

filename      = 'basketSmall';
dateStr       = '2014-05-21';
outputPath    = sprintf('../output/%s/%s',dateStr,filename);
warpFieldPath = sprintf('../output/%s/%s/%s.stw', dateStr, filename, filename);
nnfPath       = sprintf('../output/%s/%s/%s_nnf.stw', dateStr, filename, filename);
videoApath    = sprintf('../data/%s_01.mov', filename);
videoBpath    = sprintf('../data/%s_02.mov', filename);
outPath       = sprintf('../output/%s/%s/%s-matlab.stw', dateStr, filename, filename);
transPath     = sprintf('../output/%s/%s/%s_trans.stw', dateStr, filename, filename);
xformPath     = sprintf('../output/%s/%s/%s_xform.stw', dateStr, filename, filename);
flowPathA      = sprintf('../output/%s/%s/%s_flowA.stw', dateStr, filename, filename);
flowPathB      = sprintf('../output/%s/%s/%s_flowB.stw', dateStr, filename, filename);

if ~exist(sprintf('../output/%s/%s/',dateStr,filename),'dir')
    mkdir(sprintf('../output/%s/%s/',dateStr,filename));
end


uvw = loadWarpingField(warpFieldPath);

videoA = uint8(255*loadVideo(videoApath));
videoA = videoA(:,:,1:size(uvw,3),:);
videoB = uint8(255*loadVideo(videoBpath));
videoB = videoB(:,:,1:size(uvw,3),:);
[h,w,nB,~] = size(videoB);
nA = size(videoA,3);

flowA = loadWarpingField(flowPathA);
flowA = flowA(:,:,1:size(uvw,3),:);
flowB = loadWarpingField(flowPathB);
flowB = flowB(:,:,1:size(uvw,3),:);


flowBwarped = backWarp3D(flowB,uvw,size(flowA));
% flowDiff = flowA-flowBwarped;
% flowDiff = cat(4,flowDiff,zeros(h,w,nA,1));

% edgeMapA = zeros(h,w,nA,1);
% for i=1:nA
%     e = edge(squeeze(videoA(:,:,i,1)),'canny');
%     edgeMapA(:,:,i) = e;
% end
edgeMapB = zeros(h,w,nA,1);
for i=1:nA
    e = edge(squeeze(videoB(:,:,i,1)),'canny');
    edgeMapB(:,:,i) = e;
end
wField = -uvw;
warpedEdge = backWarp3D(edgeMapB,uvw,size(flowA));
% wField(:,:,:,1:2) = wField(:,:,:,1:2);

% cost = 1-edgeMapA;
% cost = single(cost);
% [warpedEdge,mask] = forwardWarping(single(edgeMapA),single(wField), cost, [1 1 1]);

alpha = 1;
newVid = videoA;
for chan = 1:size(videoA,4)
    channel = newVid(:,:,:,chan);
    idx = find(warpedEdge>0);
    channel(idx) = 255-channel(idx);
    newVid(:,:,:,chan) = channel;
end

[X,Y] = meshgrid(1:w,1:h);
step = 10;
subsX = 1:step:w;
subsY = 1:step:h;
X = X(subsY,subsX);
Y = Y(subsY,subsX);
for frame = 10%:nA
    ff = squeeze(newVid(:,:,frame,:));
    figure;
    imshow(ff)
    hold on
    u = uvw(:,:,frame,1);
    v = uvw(:,:,frame,2);
    u = u(subsY,subsX);
    v = v(subsY,subsX);
    quiver(X,Y,u,v,0)
end

exportVideo(newVid,outputPath,'overlay');
