function [F,spCount] = extractFeatures(flow,slic,dx,dy,video)
    n = size(flow,3);
    aBin = -180:20:180;
    aBin = aBin(1:end-1);
    aBinSp = 0:20:180;
    aBinSp = aBinSp(1:end-1);
    cBin = 0:20:255;
    F = [];
    spCount = zeros(n-1,1);
    for frame = 1:n-1
        s = slic(:,:,frame);
        u = flow(:,:,frame,1);
        v = flow(:,:,frame,2);
        dxf = dx(:,:,frame);
        dyf = dy(:,:,frame);
        r = video(:,:,frame,1);
        g = video(:,:,frame,2);
        b = video(:,:,frame,3);
        superpix = unique(s(:));
        spCount(frame) = length(superpix);
        feats = zeros(length(superpix),length(aBin)+length(aBinSp)+1);
        % feats = zeros(length(superpix),length(aBin)+1+length(cBin));
        for px = 1:superpix(end);
            ids = find(s==px);
            uu = u(ids);
            vv = v(ids);
            angl = atan2(vv,uu);
            angl = 180*angl/pi;
            mag = sqrt(uu.^2+vv.^2);
            histo = weighted_histogram_interp(angl,mag,aBin);
            wS = sum(histo);
            if wS>0
                histo = histo / wS;
            end
            % histo = histo/numel(ids);
            
            % rr = r(ids);
            % gg = g(ids);
            % bb = b(ids);
            % hr = weighted_histogram_interp(rr,ones(size(rr,1),1),cBin);
            % hg = weighted_histogram_interp(gg,ones(size(gg,1),1),cBin);
            % hb = weighted_histogram_interp(bb,ones(size(bb,1),1),cBin);
            % hco = cat(1,hr,hg,hb);
            % hr = hr / length(rr);

            ddx = dxf(ids);
            ddy = dyf(ids);
            angl = atan2(ddy,ddx);
            angl = abs(angl);
            angl = 180*angl/pi;
            mag = sqrt(ddx.^2+ddy.^2);
            histo2 = weighted_histogram_interp(angl,mag,aBinSp);
            wS = sum(histo2);
            if wS>0
                histo2 = histo2 / wS;
            end
            feats(px,:) = cat(1,histo,histo2,mean(mag));
            % feats(px,:) = cat(1,histo,mean(mag),hr);
        end
        F = cat(1,F,feats);
    end

end % extractFeatures
