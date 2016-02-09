% ----------------------------------------------------------------------------
% File:    weighted_histogram.m
% Author:  Michael Gharbi <gharbi@mit.edu>
% Created: 2014-05-15
% ----------------------------------------------------------------------------
% 
% 
% 
% ---------------------------------------------------------------------------%

function [h,wsum] = weighted_histogram(values, weights, bins)
    %% TODO: check bins and range agree
    nbins = length(bins);
    h = zeros(nbins,1);

    [values,bb] = sort(values);
    weights = weights(bb);
    i = 1;
    for binIdx = 1:length(bins)-1
        while  i <=length(values) && values(i)<bins(binIdx+1) 
            h(binIdx) = h(binIdx) + weights(i);
            i = i+1;
        end
    end

    wsum = sum(h);
end % weighted_histogram
