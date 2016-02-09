motion_install;

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
slicPrototypeA = sprintf('../output/%s/%s/slicA/%s-%s.dat', dateStr, filename, filename,'%04d');
slicPrototypeB = sprintf('../output/%s/%s/slicB/%s-%s.dat', dateStr, filename, filename,'%04d');

if ~exist(sprintf('../output/%s/%s/',dateStr,filename),'dir')
    mkdir(sprintf('../output/%s/%s/',dateStr,filename));
end

videoA = uint8(255*loadVideo(videoApath));
videoB = uint8(255*loadVideo(videoBpath));
if exist(flowPathA,'file')
    flowA = loadWarpingField(flowPathA);
else
    flowA = computeOpticalFlow(videoA);
    saveWarpingField(flowA,flowPathA);
end
if exist(flowPathB,'file')
    flowB = loadWarpingField(flowPathB);
else
    flowB = computeOpticalFlow(videoB);
    saveWarpingField(flowB,flowPathB);
end

[height, width, nA, ~] = size(flowA);
nB = size(flowB,3);

slicA = loadSLIC(size(flowA),slicPrototypeA);
slicB = loadSLIC(size(flowB),slicPrototypeB);

grayA = mean(videoA,4);
dxA = imfilter(grayA,[-1 1]);
dyA = imfilter(grayA,[-1 1]');
grayB = mean(videoA,4);
dxB = imfilter(grayB,[-1 1]);
dyB = imfilter(grayB,[-1 1]');

[FA,spCountA] = extractFeatures(flowA,slicA,dxA,dyA,videoA);
[FB,spCountB] = extractFeatures(flowB,slicB,dxB,dyB,videoB);

F = cat(1,FA,FB);
m = mean(F);
v = var(F);
v = sqrt(v);
F = (F-repmat(m,[size(F,1),1]))./repmat(v+eps,[size(F,1) 1]);
nClust = 30;
finalFeatA = zeros(nA-1,nClust);
finalFeatB = zeros(nB-1,nClust);
[clustId, clustCtr] = kmeans(F,nClust);

figure;plot(m)
hold on
plot(m+v,'r')
plot(m-v,'r')

readPtr = 1;
for i = 1:nA-1;
    nxt = readPtr+spCountA(i)-1;
    ids = clustId(readPtr:nxt);
    hh = hist(ids, 1:nClust);
    finalFeatA(i,:) = hh;
    readPtr = nxt;
end
for i = 1:nB-1;
    nxt = readPtr+spCountB(i)-1;
    ids = clustId(readPtr:nxt);
    hh = hist(ids, 1:nClust);
    finalFeatB(i,:) = hh;
    readPtr = nxt;
end

final = cat(1,finalFeatA,finalFeatB);
% m = mean(final);
% v = var(final);
% v = sqrt(v);
% finalFeatA = (finalFeatA-repmat(m,[size(finalFeatA,1),1]))./repmat(v+eps,[size(finalFeatA,1) 1]);
% finalFeatB = (finalFeatB-repmat(m,[size(finalFeatB,1),1]))./repmat(v+eps,[size(finalFeatB,1) 1]);

parDtw = [];
Xs = {finalFeatA', finalFeatB'};
aliDtw = dtw(Xs, [], parDtw);

P = aliDtw.P;
figure;
plot(P(:,1),P(:,2))
hold on
plot([1 P(end,1)],[1 P(end,2)],'r')
axis equal
axis([1 P(end,1) 1 P(end,2)])

uvw = zeros(height,width,nA,3);
w = P(:,2) - P(:,1);
for i=1:nA-1
    uvw(:,:,i,3) = w(i);
end
uvw(:,:,nA,3) = nB-nA;

saveWarpingField(single(uvw), warpFieldPath);
