function [out, d, other_coord] = nn_gallery(src, db,nnf,dist,coord,params_knnf)
    psz = [params_knnf.patch_size_space, params_knnf.patch_size_time];
    src_p = src(coord(1):coord(1)+psz(1)-1,...
            coord(2):coord(2)+psz(1)-1,...
            coord(3):coord(3)+psz(2)-1,:);

    knn = params_knnf.knn;

    nn_p = zeros(psz(1),knn*(psz(1)+1), psz(2),size(db,4),'uint8');
    other_coord = nnf(coord(2),coord(1),coord(3),:) + 1;
    d = dist(coord(1),coord(2),coord(3),:);
    for k = 1:knn
        c = other_coord(3*(k-1)+1:3*k);
        nn_p(:,(k-1)*(psz(1)+1)+1:k*(psz(1)+1)-1,:,: ) = ...
            db( c(1):c(1)+psz(1)-1,...
                c(2):c(2)+psz(1)-1,...
                c(3):c(3)+psz(2)-1,:);
    end

    out = cat(2,src_p,nn_p);

    other_coord = squeeze(other_coord);
    other_coord = reshape(other_coord,[3,knn])';

    d = squeeze(d);
    

    % figure;
    % subplot(121);
    % imshow(squeeze(src_p(:,:,1,:)));
    % title(sprintf('coord: %d %d %d', coord))
    % subplot(122);
    % imshow(squeeze(nn_p(:,:,1,:)));
end % nn_gallery
