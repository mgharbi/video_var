filename      = 'golfVariable';
dateStr       = '2014-05-13';
warpFieldPath = sprintf('../output/%s/%s/%s.stw', dateStr, filename, filename);
nnfPath       = sprintf('../output/%s/%s/%s_nnf.stw', dateStr, filename, filename);
videoApath    = sprintf('../data/%s_01.mov', filename);
videoBpath    = sprintf('../data/%s_02.mov', filename);
outPath       = sprintf('../output/%s/%s/%s-matlab.stw', dateStr, filename, filename);
flowApath     = sprintf('../output/%s/%s/%s-flowA.stw', dateStr, filename, filename);
flowBpath     = sprintf('../output/%s/%s/%s-flowB.stw', dateStr, filename, filename);

field = loadWarpingField(warpFieldPath);
flowA = loadWarpingField(flowApath);
flowB = loadWarpingField(flowBpath);

sz = size(field);
videoA = loadVideo(videoApath);
videoB = loadVideo(videoBpath);
regA = registerField(videoA,flowA);

fused = zeros(size(videoA));
for i = 1:sz(3)-1
    fused(:,:,i,:)= imfuse(squeeze(videoA(:,:,i,:)),squeeze(regA(:,:,i+1,:)));
end
fused = uint8(fused);

return

registered = registerField(field,flowA);

[X,Y,~] = meshgrid(1:sz(2), 1:sz(1), 1:sz(3));
regX = registerField(X,flowA);
regY = registerField(Y,flowA);

x = 70;
c = 1;
s1 = squeeze(field(x,:,:,c));
s2 = squeeze(registered(x,:,:,c));

r = squeeze(regY(x,:,:));
figure;surf(r)

figure;surf(s1)
figure;surf(s2)



f = fft(registered,[],3);
f(:,:,1,:) = 0;
ff = ifft(f,[],3);
% x = squeeze(field(10,10,:,1));
% x2 = squeeze(registered(10,10,:,1));
% plot(x)
% hold on
% plot(x2,'red')
