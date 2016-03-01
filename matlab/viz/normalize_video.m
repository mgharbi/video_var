function output = normalize_video(v)
    output = single(v);
    mini = min(output(:));
    maxi = max(output(:));

    output = (output-mini)/(maxi-mini+eps);
    output = uint8(255*output);
end % normalize_video
