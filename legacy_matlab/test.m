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

videoA = 255*loadVideo(videoApath);
videoB = 255*loadVideo(videoBpath);

sa = size(videoA);
sb = size(videoB);

uvw = loadWarpingField(warpFieldPath);
videoA = videoA(:,:,1:size(uvw,3),:);
render(videoA,videoB,uvw, [-2 -2 1], outputPath,'space2');
render(videoA,videoB,uvw, [1 1 1], outputPath,'align');
return;

% load NNF
if ~exist('nnf','var') 
    nnf = loadWarpingField(nnfPath);
    sz = size(nnf);
    patchSize = [11 11 7];
    valid_flag = ones(sz(1:3));
    valid_flag(end-(patchSize(1)-1)+1:end,:,:) = 0;
    valid_flag(:,end-(patchSize(2)-1)+1:end,:) = 0;
    valid_flag(:,:,end-(patchSize(2)-1)+1:end) = 0;
end

use_affine = true;

processIndependantFrames = 3;

switch processIndependantFrames
    case 1 % independant per frame GC, independant transforms
        uvw_trans = zeros(size(nnf));
        uvw_xform = zeros(size(nnf));
        for frame = 1:sz(3)-patchSize(3)-1
            % dominant translations
            tic;
            fprintf('-- computing dominant offsets...');
            % [~, top_uvw] = dominantTranslations2D(nnf,valid_flag,frame);
            top_uvw = [];
            t = toc;
            fprintf('%fs\n',t);

            % dominant transforms
            tic;
            fprintf('-- computing dominant transforms...');
            top_xform = dominantTransforms2D(nnf, valid_flag,use_affine,frame);
            t = toc;
            fprintf('%fs\n',t);

            % segment
            [uvw, uvw_t, uvw_x] = segmentUVW_2D(videoA, videoB, top_uvw, top_xform, frame,use_affine);
            uvw_trans(:,:,frame,:) = uvw_t;
            uvw_xform(:,:,frame,:) = uvw_x;

        end
    case 2 % independant GC, global transform
        uvw_trans = zeros(size(nnf));
        uvw_xform = zeros(size(nnf));
        % dominant translations
        tic;
        fprintf('-- computing dominant offsets...');
        [~, top_uvw] = dominantTranslations(nnf,valid_flag);
        % top_uvw = [];
        t = toc;
        fprintf('%fs\n',t);

        % dominant transforms
        tic;
        fprintf('-- computing dominant transforms...');
        top_xform = dominantTransforms(nnf, valid_flag,use_affine);
        t = toc;

        fprintf('%fs - %d transforms\n',t, length(top_xform));
        fprintf('%fs - %d translations\n',t, length(top_uvw));
        for frame = 1:sz(3)
            fprintf('%02d/%02d\n',frame,sz(3));

            % segment
            [uvw, uvw_t, uvw_x] = segmentUVW_2D(videoA, videoB, top_uvw, top_xform, frame,use_affine);
            if ~isempty(uvw_t)
                uvw_trans(:,:,frame,:) = uvw_t;
            end
            if ~isempty(uvw_x)
                uvw_xform(:,:,frame,:) = uvw_x;
            end
        end
    otherwise
        % dominant translations
        tic;
        fprintf('-- computing dominant offsets...');
        [~, top_uvw] = dominantTranslations(nnf,valid_flag);
        % top_uvw = [];
        t = toc;
        fprintf('%fs\n',t);

        % dominant transforms
        tic;
        fprintf('-- computing dominant transforms...');
        top_xform = dominantTransforms(nnf, valid_flag,use_affine);
        t = toc;

        fprintf('%fs - %d transforms\n',t, length(top_xform));
        fprintf('%fs - %d translations\n',t, length(top_uvw));

        % segment
        [uvw, uvw_trans, uvw_xform] = segmentUVW(videoA, videoB, top_uvw, top_xform, use_affine);
end

saveWarpingField(single(uvw_trans), transPath);
saveWarpingField(single(uvw_xform), xformPath);

% export processed flow
