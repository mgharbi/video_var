function [bestA, validA, validB] = ransacStip(posA, posB)
    n_ransac_it = 150;
    n_pt_ransac = 4;

    nA = size(posA,1);
    nB = size(posB,1);

    X1 = posA(:,1);
    Y1 = posA(:,2);
    T1 = posA(:,3);
    X2 = posB(:,1);
    Y2 = posB(:,2);
    T2 = posB(:,3);

    nAgree = 0;
    bestA = zeros(3,4);
    for ransac_iter = 1:n_ransac_it
        rnd_prm = randperm(numel(X1));
        rnd_prm2 = randperm(numel(X2));

        % setup matching point pairs
        src = zeros(4,n_pt_ransac);
        target = zeros(3,n_pt_ransac);
        for i = 1:n_pt_ransac
            src(1,i) = X1(rnd_prm(i));
            src(2,i) = Y1(rnd_prm(i));
            src(3,i) = T1(rnd_prm(i));
            src(4,i) = 1;

            target(1,i) = X2(rnd_prm2(i));
            target(2,i) = Y2(rnd_prm2(i));
            target(3,i) = T2(rnd_prm2(i));
        end
        % compute affine transform
        A = mrdivide(target*src', (src*src'));


        % compute residual and agreeing points
        projX2 = A(1,1)*X1 + A(1,2)*Y1 + A(1,3)*T1 + A(1,4);
        projY2 = A(2,1)*X1 + A(2,2)*Y1 + A(2,3)*T1 + A(2,4);
        projT2 = A(3,1)*X1 + A(3,2)*Y1 + A(3,3)*T1 + A(3,4);


        thresh = 2;
        validX = abs(X2-projX2); validX = (validX<thresh);
        validY = abs(Y2-projY2); validY = (validY<thresh);
        validT = abs(T2-projT2); validT = (validT<thresh);
        valid = validX.*validY.*validT;
        valid = find(valid);

        % record transform with sufficient agreement
        if length(valid) > nAgree
            % refine xform estimation
            valid = valid(1:min(length(valid),50000));
            vX1 = X1(valid);
            vY1 = Y1(valid);
            vT1 = T1(valid);
            vX2 = X2(valid);
            vY2 = Y2(valid);
            vT2 = T2(valid);

            src    = cat(1,vX1',vY1',vT1',ones(size(vX1')));
            target = cat(1,vX2',vY2',vT2');
            bestA      = mrdivide(target*src', (src*src'));
            nAgree = length(valid);
            validA = posA(valid,:);
            validB = posB(valid,:);
        end
            fprintf('%03d - %04d matches\n', ransac_iter, nAgree)
    end % ransac_iter
end % ransacStip
