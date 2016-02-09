
% ----------------------------------------------------------------------------
% File:    dominantTransforms.m
% Author:  Michael Gharbi <gharbi@mit.edu>
% Created: 2014-03-19
% ----------------------------------------------------------------------------
% 
% 
% 
% ---------------------------------------------------------------------------%


function top_xform = dominantTransforms2D(nnf, valid_flag,use_affine,frame)
    % Params
    [height,width,nF,~]=size(nnf);
    n_ransac_it = 150;
    count_threshold = 10000;
    if use_affine
        n_pt_ransac = 4;
    else
        n_pt_ransac = 3;
    end

    [X1, Y1] = meshgrid(1:width, 1:height);
    T1 = frame*ones(height, width);
    X2 = X1 + nnf(:,:,frame,1);
    Y2 = Y1 + nnf(:,:,frame,2);
    T2 = T1 + nnf(:,:,frame,3);

    if exist('valid_flag','var') && ~isempty(valid_flag)
        valid_flag = valid_flag(:,:,frame);
        ind = find(valid_flag==1);
        X1 = X1(ind); Y1 = Y1(ind); T1 = T1(ind);
        X2 = X2(ind); Y2 = Y2(ind); T2 = T2(ind);
    end

    top_xform = [];
    valid_index = find(valid_flag);
    if length(valid_index) < count_threshold
        fprintf('not enough points for reliable transform estimation');
        return;
    end

    transform_id = 1;
    for ransac_iter = 1:n_ransac_it
        rnd_prm = randperm(numel(X1));

        if use_affine
            % setup matching point pairs
            src = zeros(4,n_pt_ransac);
            target = zeros(3,n_pt_ransac);
            for i = 1:n_pt_ransac
                src(1,i) = X1(rnd_prm(i));
                src(2,i) = Y1(rnd_prm(i));
                src(3,i) = T1(rnd_prm(i));
                src(4,i) = 1;

                target(1,i) = X2(rnd_prm(i));
                target(2,i) = Y2(rnd_prm(i));
                target(3,i) = T2(rnd_prm(i));
            end
            % compute affine transform
            A = mrdivide(target*src', (src*src'));

            % compute residual and agreeing points
            projX2 = A(1,1)*X1 + A(1,2)*Y1 + A(1,3)*T1 + A(1,4);
            projY2 = A(2,1)*X1 + A(2,2)*Y1 + A(2,3)*T1 + A(2,4);
            projT2 = A(3,1)*X1 + A(3,2)*Y1 + A(3,3)*T1 + A(3,4);

        else
            A = zeros(n_pt_ransac*2,4);b = zeros(n_pt_ransac*2,1);
            t_src = ones(2,n_pt_ransac);
            t_target = zeros(1,n_pt_ransac);
            for i = 1:n_pt_ransac
                tmpA = [X1(rnd_prm(i)) -Y1(rnd_prm(i)) 1 0;...
                    Y1(rnd_prm(i)) X1(rnd_prm(i)) 0 1];
                tmpb = [X2(rnd_prm(i)); Y2(rnd_prm(i))];
                A(i*2-1:i*2,:) = tmpA; b(i*2-1:i*2) = tmpb;
                t_src(1,i) = T1(rnd_prm(i));
                t_target(1,i) = T2(rnd_prm(i));
            end
            tmphomo = A\b;
            tmphomo = [tmphomo(1) -tmphomo(2) tmphomo(3);...
                tmphomo(2) tmphomo(1) tmphomo(4)];

            % timeParams = mrdivide(t_target*t_src',t_src*t_src');
            timeParams = [1,0];

            % computing residual
            projX2 = tmphomo(1,1) * X1 + tmphomo(1,2) * Y1 + tmphomo(1,3);
            projY2 = tmphomo(2,1) * X1 + tmphomo(2,2) * Y1 + tmphomo(2,3);
            projT2 = timeParams(1)* T1 + timeParams(2);
        end

        thresh = 1;
        validX = abs(X2-projX2); validX = (validX<thresh);
        validY = abs(Y2-projY2); validY = (validY<thresh);
        validT = abs(T2-projT2); validT = (validT<thresh);
        valid = validX.*validY.*validT;
        valid = find(valid);



        % record transform with sufficient agreement
        if length(valid) > count_threshold
            fprintf( '%03d - valid: %04d\n', ransac_iter, length(valid));
            top_xform(transform_id).count = length(valid);

            % refine xform estimation
            valid = valid(1:min(length(valid),50000));
            vX1 = X1(valid);
            vY1 = Y1(valid);
            vT1 = T1(valid);
            vX2 = X2(valid);
            vY2 = Y2(valid);
            vT2 = T2(valid);
            if use_affine
                src = cat(1,vX1',vY1',vT1',ones(size(vX1')));
                target = cat(1,vX2',vY2',vT2');
                A = mrdivide(target*src', (src*src'));
                top_xform(transform_id).matrix = A;
            else 
                count = length(vX1);
                A = zeros(2*count,4); b = zeros(2*count,1);
                t_src = ones(2,count);
                t_target = zeros(1,count);
                for j = 1:count
                    tmpA = [vX1(j) -vY1(j) 1 0;...
                            vY1(j)  vX1(j) 0 1];
                    tmpb = [vX2(j); vY2(j)];
                    A(j*2-1:j*2,:) = tmpA; b(j*2-1:j*2) = tmpb;
                    t_src(1,j) = T1(rnd_prm(j));
                    t_target(1,j) = T2(rnd_prm(j));
                end
                tmphomo = A\b;
                tmphomo = [tmphomo(1) -tmphomo(2) tmphomo(3);...
                           tmphomo(2) tmphomo(1) tmphomo(4)];
                timeParams = mrdivide(t_target*t_src',t_src*t_src');
                top_xform(transform_id).matrix = tmphomo;
                top_xform(transform_id).timeParams = timeParams;
            end

            transform_id = transform_id+1;
        else
            fprintf( '%03d -        %04d\n', ransac_iter, length(valid));
        end

    end % ransac_iter
    fprintf( '%03d - valid xform with count > %d\n', length(top_xform), n_ransac_it);

end % dominantTransforms
