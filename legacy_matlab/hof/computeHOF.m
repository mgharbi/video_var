% ----------------------------------------------------------------------------
% File:    computeHOF.m
% Author:  Michael Gharbi <gharbi@mit.edu>
% Created: 2014-05-19
% ----------------------------------------------------------------------------
% 
% 
% 
% ---------------------------------------------------------------------------%

function feats = computeHOF(flow,orientationStep,h_sBin,v_sBin)
    [h,w,nF,~] = size(flow);
    if nargin < 2
        orientationStep = 10;
    end
    if nargin < 3
        h_sBin = 30;
    end
    if nargin < 4
        v_sBin = h_sBin;
    end
    aBin = -180:orientationStep:180;

    % h_sBin = w;
    % v_sBin = h;

    h_bins = 2*floor(w/h_sBin) - 1;
    v_bins = 2*floor(h/v_sBin) - 1;

    featsPerBin = length(aBin);
    feats = zeros(nF,h_bins*v_bins*featsPerBin);

    addFrames = 0;
    for frame = 1+addFrames:nF-addFrames
        u = squeeze(flow(:,:,frame-addFrames:frame+addFrames,1));
        v = squeeze(flow(:,:,frame-addFrames:frame+addFrames,2));
        angl = atan2(v,u);
        angl = 180*angl/pi;
        mag = sqrt(u.^2+v.^2);
        
        for hIdx = 0:h_bins-1
            for vIdx = 0:v_bins-1
                hh = hIdx*(h_sBin/2)+1:hIdx*(h_sBin/2)+h_sBin;
                vv = vIdx*(v_sBin/2)+1:vIdx*(v_sBin/2)+v_sBin;

                binIdx = hIdx*v_bins+vIdx;

                l_mag = reshape(mag(vv,hh,:),[],1);
                l_angl = reshape(angl(vv,hh,:),[],1);
                % histo = weighted_histogram(l_angl,l_mag,aBin);
                histo = weighted_histogram_interp(l_angl,l_mag,aBin);
                
                % normalize 
                wHisto = sum(histo);
                if wHisto > 0
                    % histo = histo/wHisto;
                end
                feats(frame,binIdx*featsPerBin+1:(binIdx+1)*featsPerBin) = histo(:);
            end
        end
        % [maxes,xMax] = max(mag);
        % [~,yMax] = max(maxes);
        % xMax = xMax(yMax);
        % feats(frame,end-1:end) = [xMax, yMax];
    end
end % computeHOF
