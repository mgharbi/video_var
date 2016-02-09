addpath(genpath(pwd));

filename      = 'newtonVariable';
dateStr       = '2014-05-22';
warpFieldPath = sprintf('../output/%s/%s/%s.stw', dateStr, filename, filename);
nnfPath       = sprintf('../output/%s/%s/%s_nnf.stw', dateStr, filename, filename);
videoApath    = sprintf('../data/%s_01.mov', filename);
videoBpath    = sprintf('../data/%s_02.mov', filename);
outPath       = sprintf('../output/%s/%s/%s-matlab.stw', dateStr, filename, filename);
transPath     = sprintf('../output/%s/%s/%s_trans.stw', dateStr, filename, filename);
xformPath     = sprintf('../output/%s/%s/%s_xform.stw', dateStr, filename, filename);
flowPathA     = sprintf('../output/%s/%s/%s_flowA.stw', dateStr, filename, filename);
flowPathB     = sprintf('../output/%s/%s/%s_flowB.stw', dateStr, filename, filename);

if ~exist(sprintf('../output/%s/%s/',dateStr,filename),'dir')
    mkdir(sprintf('../output/%s/%s/',dateStr,filename));
end


if exist(flowPathA,'file')
    flowA = loadWarpingField(flowPathA);
else
    videoA = uint8(255*loadVideo(videoApath));
    flowA = computeOpticalFlow(videoA);
    saveWarpingField(flowA,flowPathA);
end
if exist(flowPathB,'file')
    flowB = loadWarpingField(flowPathB);
else
    videoB = uint8(255*loadVideo(videoBpath));
    flowB = computeOpticalFlow(videoB);
    saveWarpingField(flowB,flowPathB);
end

[height, width, nA, ~] = size(flowA);
nB = size(flowB,3);

% Acceleration instead?
% flowA(:,:,1:end-1,:) = flowA(:,:,1:end-1,:) - flowA(:,:,2:end,:);
% flowB(:,:,1:end-1,:) = flowB(:,:,1:end-1,:) - flowB(:,:,2:end,:);

signal_a = computeHOF(flowA(:,:,1:end-1,:));
signal_b = computeHOF(flowB(:,:,1:end-1,:));

parDtw = [];
Xs = {signal_a', signal_b'};
aliDtw = dtw(Xs, [], parDtw);
% parCtw = st('debg', 'n');
% parCca = st('d', .8, 'lams', .6); % CCA: reduce dimension to keep at least 0.8 energy, set the regularization weight to .6
% aliCtw = ctw(Xs, aliDtw, [], parCtw, parCca, parDtw);


P = aliDtw.P;
figure;
plot(P(:,1),P(:,2))
% P2 = aliCtw.P;
% hold on 

% relative coordinate
uvw = zeros(height,width,nA,3);
w = P(:,2) - P(:,1);
for i=1:nA-1
    uvw(:,:,i,3) = w(i);
end
uvw(:,:,nA,3) = nB-nA;


saveWarpingField(single(uvw), warpFieldPath);

% videoA = uint8(255*loadVideo(videoApath));
% videoB = uint8(255*loadVideo(videoBpath));
% warpedB = uint8(backWarp3D(videoB,uvw,size(videoA)));
% uvw2 = computeSpatialMatch(videoA,warpedB);
% uvw(:,:,:,1) = uvw2(:,:,:,1);
% uvw(:,:,:,2) = uvw2(:,:,:,2);
% saveWarpingField(single(uvw ), warpFieldPath);

% frame = 30;
% figure;
% imshow(imfuse(squeeze(videoA(:,:,frame,:)), squeeze(warpedB(:,:,frame,:))));

