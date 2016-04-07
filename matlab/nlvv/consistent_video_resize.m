function out = consistent_video_resize(in, src_scale, dst_scale, finest_res)
% in: input video
% src_scale: triplet giving the source size ratio compared to the finest resolution
% dst_scale: triplet giving the destination size ratio compared to the finest resolution
% finest_res: triplet giving the pixel resolution of the finest scale

h = finest_res(1);
w = finest_res(2);
nF = finest_res(3);

[hi,wi,nFi,nCi] = size(in);

src_step = 1./src_scale;
dst_step = 1./dst_scale;

max_dst_y = 1 + floor( (h -1)/dst_step(1) ) * dst_step(1);
max_dst_x = 1 + floor( (w -1)/dst_step(2) ) * dst_step(2);
max_dst_t = 1 + floor( (nF-1)/dst_step(3) ) * dst_step(3);

padsize_y = ceil(max(1 ,h-max_dst_y));
padsize_x = ceil(max(1, w-max_dst_x));
padsize_t = ceil(max(1,nF-max_dst_t));

in = padarray(in, [padsize_y, padsize_x, padsize_t, 0], 'replicate', 'post');

% Sampling coordinates in the source
[src_x, src_y, src_t] = meshgrid(...
    src_step(2):src_step(2):(w +padsize_x*src_step(2)),...
    src_step(1):src_step(1):(h +padsize_y*src_step(1)),...
    src_step(3):src_step(3):(nF+padsize_t*src_step(3))...
);

% Sampling coordinates in the target
[dst_x, dst_y, dst_t] = meshgrid(...
    dst_step(2):dst_step(2):(w +padsize_x*dst_step(2)),...
    dst_step(1):dst_step(1):(h +padsize_y*dst_step(1)),...
    dst_step(3):dst_step(3):(nF+padsize_t*dst_step(3))...
);

% AA Prefilter
sampling_ratio = dst_scale ./ src_scale;
sampling_ratio
if min(sampling_ratio) < 1
    sigma = sqrt(-2*log(exp(-1))) ./ (sampling_ratio * pi);
    for c = 1:nCi
        in(:,:,:,c) = imgaussfilt3(in(:,:,:,c), sigma);
    end
end

% Interpolate
out = zeros([size(dst_x), nCi]);
for c = 1:nCi
    out(:,:,:,c) = interp3(src_x,src_y,src_t, in(:,:,:,c) , dst_x, dst_y, dst_t ,'spline');
end

% Remove padding
out =  out(1:end-padsize_y, 1:end-padsize_x,1:end-padsize_t,:);

end % consistent_image_resize
