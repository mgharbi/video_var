addpath(genpath(pwd));

filename      = 'derailleur';
dateStr       = '2014-05-20';
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


videoA = (255*loadVideo(videoApath));
videoB = (255*loadVideo(videoBpath));
sa = size(videoA);
sb = size(videoB);
uvw = loadWarpingField(warpFieldPath);
videoA = videoA(:,:,1:size(uvw,3),:);

if exist(flowPathA,'file')
    flowA = loadWarpingField(flowPathA);
else
    flowA = computeOpticalFlow(uint8(videoA));
    saveWarpingField(flowA,flowPathA);
end
% if exist(flowPathB,'file')
%     flowB = loadWarpingField(flowPathB);
% else
%     flowB = computeOpticalFlow(videoB);
%     saveWarpingField(flowB,flowPathB);
% end

render(videoA,videoB,uvw, [-2 -2 1], outputPath,'space2',flowA);
