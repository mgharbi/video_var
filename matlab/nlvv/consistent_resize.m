function [video_out] = consistent_resize(video_in, new_size)

    [h,w,nF,c] = size(video_in);
    video_out = zeros(new_size);
    for f = 1:nF
        video_out(:,:,f,:) = imresize(squeee(video_in(:,:,f,:)));
    end

    % TODO
    % Anti-aliasing filter
    % Sampling coordinates

end % consistent_resize
