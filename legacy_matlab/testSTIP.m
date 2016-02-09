close all;
addpath(genpath(pwd));

filename      = 'newtonVariable';
dateStr       = '2014-05-21';
warpFieldPath = sprintf('../output/%s/%s/%s.stw', dateStr, filename, filename);
nnfPath       = sprintf('../output/%s/%s/%s_nnf.stw', dateStr, filename, filename);
videoApath    = sprintf('../data/%s_01.mov', filename);
videoBpath    = sprintf('../data/%s_02.mov', filename);
stipApath     = sprintf('../data/stip/%s_01.stip', filename);
stipBpath     = sprintf('../data/stip/%s_02.stip', filename);
outPath       = sprintf('../output/%s/%s/%s-matlab.stw', dateStr, filename, filename);
transPath     = sprintf('../output/%s/%s/%s_trans.stw', dateStr, filename, filename);
xformPath     = sprintf('../output/%s/%s/%s_xform.stw', dateStr, filename, filename);

if ~exist(sprintf('../data/stip/%s_01.stip',filename),'file')
    cmd = '../bin/stip -f ../data/%s_%02d.mov -o ../data/stip/%s_%02d.stip';
    for i =1:2
        system(sprintf(cmd,filename, i, filename, i))
    end
end


videoA = 255*loadVideo(videoApath);
videoB = 255*loadVideo(videoBpath);

stipA = loadStip(stipApath);
stipB = loadStip(stipBpath);

nA = size(stipA,1);
nB = size(stipB,1);

szA = size(videoA);
szB = size(videoB);

xA = stipA(:,1:3);
xB = stipB(:,1:3);

tA = xA(:,3);
tB = xB(:,3);

nF = max(szA(3),szB(3));

signal_a = zeros(nA,size(stipA,2));
signal_b = zeros(nB,size(stipB,2));

for i=1:nA
    idx = find(tA==i);
    st = stipA(idx,:);
    signal_a(i,:) = mean(st,1);
end
for i=1:nB
    idx = find(tB==i);
    st = stipB(idx,:);
    signal_b(i,:) = mean(st,1);
end
signal_a = signal_a(:,4:end);
signal_b = signal_b(:,4:end);

parDtw = [];
Xs = {signal_a', signal_b'};
aliDtw = dtw(Xs, [], parDtw);
P = aliDtw.P;

figure;
plot(P(:,1),P(:,2))
