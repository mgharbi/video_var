addpath(genpath(pwd));

filename      = 'golfShort';
dateStr       = '2014-04-24';
warpFieldPath = sprintf('../output/%s/%s/%s.stw', dateStr, filename, filename);
nnfPath       = sprintf('../output/%s/%s/%s_nnf.stw', dateStr, filename, filename);
videoApath    = sprintf('../data/%s_01.mov', filename);
videoBpath    = sprintf('../data/%s_02.mov', filename);
outPath       = sprintf('../output/%s/%s/%s-matlab.stw', dateStr, filename, filename);
transPath     = sprintf('../output/%s/%s/%s_trans.stw', dateStr, filename, filename);
xformPath     = sprintf('../output/%s/%s/%s_xform.stw', dateStr, filename, filename);

if ~exist('videoA','var') 
    videoA = 255*loadVideo(videoApath);
end
if ~exist('videoB','var') 
    videoB = 255*loadVideo(videoBpath);
end

frame = size(videoA,3);
I1  = (squeeze(videoA(:,:,frame,:)));
I2  = (squeeze(videoB(:,:,frame,:)));
I1 = uint8(I1);
I2 = uint8(I2);
uv  = estimate_flow_interface(I1,I2, 'hs');
I2w = uint8(backWarp2D(double(I2),uv));

figure; imshow(flowToColor(uv));
figure; imshow(I2w);
figure; imshow(imfuse(I1,I2w));


% NNF
csh_wid = 16;
[uv1,flag1] = CSH_nn_flow_base(I1,I2,[],csh_wid,1);
uuu = uv1;
[histmat_csh, top_uv2] = Dominant_Offset(uv1,flag1);
%% top homography computation by SIFT
top_homo = Dominant_Transform(uv1);
%% motion segmentation
[uv,uv1,uv2] = knn_flow(I1,I2,top_uv2,top_homo);
img = flowToColor(uv);
%% continuous refinement
uvo = estimate_flow_interface2(I1,I2, 'classic+nl-fast', [], uv);

I2w2 = uint8(backWarp2D(double(I2),uvo));

figure; imshow(flowToColor(uvo));
figure; imshow(I2w2);
figure; imshow(imfuse(I1,I2w2));

figure; imshow(imfuse(I1,I2));
