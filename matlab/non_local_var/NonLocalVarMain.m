function [img_regular, img_warped, ux, uy] = NonLocalVarMain(img, param, img_regular, ux, uy)
% -------------------------------------------------------------------------
% NonLocalVarMain (img, param, img_regular, ux, uy) the main function the
% implements the non local variations iterative alg.
% Input args:
%         (1) img - input image
%         (2) param - a structure of parameters (see NonLocalVarMultiScale)
%         (3) img_regular - an initialization for the regular image (used
%         when propImgRegular=1)
%         (4) ux,uy - an initialization for the flow field
% Output args:
%          (1) img_regular - the 'ideal' computed image
%          (2) img_warped - the 'corrected' image computed by warping the
%          input
%          (3) ux, uy - the computed warping field
% -------------------------------------------------------------------------
% parsing the input parameters
if(~isfield(param, 'PatchSize'))
    PatchSize = [15,15];
else
    PatchSize = param.PatchSize;
end

if(~isfield(param, 'alpha'))
    alpha = 0.04; % relative weight of the smoothness term of the flow
else
    alpha = param.alpha;
end

if(~isfield(param, 'lambda'))
    lambda = 20;
else
    lambda = param.lambda;
end

% Number of outer iterations
if(~isfield(param, 'NumIterOuter'))
    NumIterOuter = 10;
else
    NumIterOuter = param.NumIterOuter;
end

% Number of inner iterations
if(~isfield(param, 'NumIterInner'))
    NumIterInner = 5;
else
    NumIterInner = param.NumIterInner;
end

% Number of NN
if(~isfield(param, 'NumNN'))
    NumNN = 20;
else
    NumNN = param.NumNN;
end

if ~isfield(param, 'propImgRegular')
    propImgRegular = 0 ;
else
    propImgRegular = param.propImgRegular;
end


img = im2double(img);
[Q,R,K] = size(img);
figure, imshow(img), title('Input image')
% initalization
if ~exist('ux', 'var') || ~exist('uy', 'var') || isempty(ux) || isempty(uy)
    ux = zeros(Q,R);
    uy = zeros(Q,R);
end


[img_warped,mask] = warpImage(img, ux, uy);
img_warped_prev = [];

if  ~propImgRegular || ~exist('img_regular', 'var') || isempty(img_regular)
    img_regular = img_warped;
end

% Optical flow parameters
ratio = 0.75;
minWidth = min(20, size(img, 2));
nOuterFPIterations = 3;
nInnerFPIterations = 4;
nSORIterations = 30;
para = [alpha,ratio,minWidth,nOuterFPIterations,nInnerFPIterations,nSORIterations];

ann = [];
 disp(['Outer iter:' num2str(NumIterOuter) ' Inner iter:' num2str(NumIterInner)]);
parfor_progress(NumIterOuter);
for i = 1:NumIterOuter
    [img_regular,ann] = UpdateImage(img_warped, img_regular, img_regular, PatchSize, NumIterInner, NumNN, lambda, ann);
    [ux,uy] = Coarse2FineTwoFramesFinal(img_regular,img,para, ux, uy, 1e-3,1e-6);
    ux(1,:) = 0; uy(1,:) = 0;
    ux(end,:) = 0; uy(end,:) = 0;
    ux(:,1) = 0; uy(:,1) = 0;
    ux(:,end) = 0; uy(:,end) = 0;
    [ux, uy] = FlowConsistency2(ux, uy, 50);
    
    [img_warped,mask] = warpImage(img, ux, uy);
    if(i> 1 & sqrt(mean((img_warped_prev(:)-img_warped(:)).^2)) < 1e-5)
        break;
    else
        img_warped_prev = img_warped;
    end
   parfor_progress;
end

parfor_progress(0);

function [img_new, ann] = UpdateImage(WarpedImage, img, img_DB, PatchSize, NumIter, NumNN, lambda, ann)
% UpdateImage function that implements the Image Update step
% (transformation is fixed)

h = 0.1; %Bandwidth paramter of NN weights kernel
beta = 1 / h^2;
if(NumNN==1)
    NumNNin = [];
else
    NumNNin = NumNN;
end

img_new = img;


for i = 1:NumIter
    
    % run PatchMatch
    ann  = nnmex(img_new,img_DB, 'cputiled', PatchSize, [], [], [], [], [], 12, [], [], [], [], [], NumNNin, []);
    Dist = double(squeeze(ann(:,:,3,:)))/65025/3;
    
    % compute weights
    W = Dist;
    W = exp(-0.5*W/prod(PatchSize)/h^2);
    W = bsxfun(@rdivide, W, sum(W, 3));
    
    img_new = ind2ImAvg_mex(im2double(img_DB), int32(squeeze(ann(:,:,1,:))+1), int32(squeeze(ann(:,:,2,:))+1), W,PatchSize(1), PatchSize(2));
    
    PsiD  = 1./sqrt(mean((WarpedImage - img_new).^2, 3) + 1e-6);
    PsiD = repmat(PsiD,[1,1,3]);
    img_new = (lambda*PsiD.*WarpedImage +  beta * img_new)./ (lambda.*PsiD + beta );
    
    
    if(i> 1 & mean(abs(img_prev(:)-img_new(:))) < 5e-4)
        break;
    else
        img_prev = img_new;
    end
    
end
