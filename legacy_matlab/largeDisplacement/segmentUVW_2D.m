% ----------------------------------------------------------------------------
% File:    segmentUVW.m
% Author:  Michael Gharbi <gharbi@mit.edu>
% Created: 2014-03-19
% ----------------------------------------------------------------------------
% 
% 
% 
% ---------------------------------------------------------------------------%


function [uvw, uvw_t, uvw_xform] = segmentUVW_2D(videoA, videoB, top_uvw, top_xform, frame,use_affine)
    amp_factor = 100;
    lambda     = 6;

    uvw       = [];
    uvw_t     = [];
    uvw_xform = [];

    [height, width, ~, ~] = size(videoA);

    Ktrans = size(top_uvw,1);

    % Keep the top transforms
    maxXforms = 40;
    Kxform = min(length(top_xform),maxXforms);
    if length(top_xform) > maxXforms
        count = [top_xform(:).count];
        [~,idx] = sort(count,'descend');
        top_xform = top_xform(idx(1:maxXforms));
    end

    % Dominant translational field
    fprintf('-- segmenting translations\n');
    if Ktrans == 1
        fprintf('\tonly one label, skipping translation.\n')
        uvw_t = ones(height,width,3);
        uvw_t(:,:,1) = uvw_t(:,:,1)*top_uvw(1,1);
        uvw_t(:,:,2) = uvw_t(:,:,2)*top_uvw(1,2);
        uvw_t(:,:,3) = uvw_t(:,:,3)*top_uvw(1,3);
    elseif Ktrans > 0
        [x,y] = meshgrid(1:width, 1:height);
        t = frame*ones(size(x));
        error_map = zeros(Ktrans, height*width, 'int32');

        % Build error maps
        fprintf('   - building error maps...\n');
        for i = 1 : Ktrans
            fprintf('     %02d/%02d. ', i, Ktrans);
            tic;
            curr_uvw = top_uvw(i,:);

            x2 = x + curr_uvw(1);
            y2 = y + curr_uvw(2);
            t2 = t + curr_uvw(3);


            warped = zeros(height, width, 3);

            for l = 1:3
                warped(:,:,l) = interp3(videoB(:,:,:,l),x2,y2,t2);
            end
            warped(isnan(warped)) = 0;

            t_error_map = (squeeze(videoA(:,:,frame,:))-warped).^2;
            t_error_map = sum(t_error_map,3);
            t_error_map = sqrt(t_error_map);

            error_map(i,:) = amp_factor*t_error_map(:)';
            time = toc;
            fprintf('%fs\n',time);
        end
        clear t_error_map;

        fprintf('   - building cost matrix...');
        tic;
        cost_matrix = zeros(Ktrans,Ktrans,'int32');
        for i = 1:Ktrans
            curr_uvw_1 = top_uvw(i,1:3);
            for j = i:Ktrans
                % TODO: L1 norm here?
                curr_uvw_2 = top_uvw(j,1:3);
                val = ( (curr_uvw_1-curr_uvw_2).^2 );
                val = sqrt(val);
                val = sum(val);
                val = lambda*val;
                val = amp_factor * val;
                val = round(val/10);
                cost_matrix(i,j) = val;
                cost_matrix(j,i) = cost_matrix(i,j);
            end
        end
        time = toc;
        fprintf('done. %fs\n',time);

        h = GCO_Create(height*width,Ktrans);
        GCO_SetDataCost(h,error_map); % data term
        GCO_SetSmoothCost(h,cost_matrix);

        i_x_neighbor = sub2ind([height, width],...
                        y(:,1:end-1,:),...
                        x(:,1:end-1,:));
        j_x_neighbor = sub2ind([height, width],...
                        y(:,2:end,:),...
                        x(:,2:end,:));
        i_y_neighbor = sub2ind([height, width],...
                        y(1:end-1,:,:),...
                        x(1:end-1,:,:));
        j_y_neighbor = sub2ind([height, width],...
                        y(2:end,:,:),...
                        x(2:end,:,:));

        is = cat(1,i_x_neighbor(:), i_y_neighbor(:));
        js = cat(1,j_x_neighbor(:), j_y_neighbor(:));

        img = squeeze(videoA(:,:,frame,:));
        [m, n, ~] = size(img);
        [weights_x, weights_y] = weighted_derivative_ops_color(m, n, img, 3, .01, 0);
        weights_x = round(weights_x*10); weights_y = round(weights_y*10);
        weights = cat(1, weights_x, weights_y);

        S = sparse(is,js,weights,height*width, height*width);
        GCO_SetNeighbors(h,S);

        % optimization
        fprintf('   - GCO...');
        tic;
        GCO_Expansion(h); 
        uvw_label=GCO_GetLabeling(h);
        uvw_label=reshape(uvw_label,[height width]);
        time = toc;
        fprintf('done. %fs\n',time);


        u = zeros(height,width);
        v = zeros(height,width);
        w = zeros(height,width);
        for i=1:Ktrans
            tmp_l = find(uvw_label==i);
            u(tmp_l) = top_uvw(i,1);
            v(tmp_l) = top_uvw(i,2);
            w(tmp_l) = top_uvw(i,3);
        end
        uvw_t = cat(3,u,v,w);
        GCO_Delete(h);
    end % translational field


    if Kxform == 1
        fprintf('\tonly one label, skipping homography.\n')
        uvw_xform = ones(height,width,3);
        [x,y] = meshgrid(1:width, 1:height);
        t = frame*ones(size(x));
        if use_affine
            A = top_xform(1).matrix;
            x2 = A(1,1)*x + A(1,2)*y + A(1,3)*t + A(1,4);
            y2 = A(2,1)*x + A(2,2)*y + A(2,3)*t + A(2,4);
            t2 = A(3,1)*x + A(3,2)*y + A(3,3)*t + A(3,4);
        else
            tmphomo = top_xform(1).matrix;
            timeParams = top_xform(1).timeParams;
            x2 = tmphomo(1,1) * x + tmphomo(1,2) * y + tmphomo(1,3);
            y2 = tmphomo(2,1) * x + tmphomo(2,2) * y + tmphomo(2,3);
            t2 = timeParams(1)* t + timeParams(2);
        end

        uvw_xform(:,:,1) = x2-x;
        uvw_xform(:,:,2) = y2-y;
        uvw_xform(:,:,3) = t2-t;
    elseif Kxform > 0
        [x,y] = meshgrid(1:width, 1:height);
        t = frame*ones(size(x));
        error_map = zeros(Kxform, height*width, 'int32');

        % Build error maps
        fprintf('   - building error maps...\n');
        uuu = zeros(Kxform,height*width);
        vvv = zeros(Kxform,height*width);
        www = zeros(Kxform,height*width);
       for i = 1 : Kxform
            fprintf('     %02d/%02d. ', i, Kxform);
            tic;

            if use_affine
                A = top_xform(i).matrix;
                x2 = A(1,1)*x + A(1,2)*y + A(1,3)*t + A(1,4);
                y2 = A(2,1)*x + A(2,2)*y + A(2,3)*t + A(2,4);
                t2 = A(3,1)*x + A(3,2)*y + A(3,3)*t + A(3,4);
            else
                tmphomo = top_xform(i).matrix;
                timeParams = top_xform(i).timeParams;
                x2 = tmphomo(1,1) * x + tmphomo(1,2) * y + tmphomo(1,3);
                y2 = tmphomo(2,1) * x + tmphomo(2,2) * y + tmphomo(2,3);
                t2 = timeParams(1)* t + timeParams(2);
            end

            uuu(i,:) = x2(:)-x(:);
            vvv(i,:) = y2(:)-y(:);
            www(i,:) = t2(:)-t(:);

            warped = zeros(height, width, 3);

            for l = 1:3
                warped(:,:,l) = interp3(videoB(:,:,:,l),x2,y2,t2);
            end
            warped(isnan(warped)) = 0;

            t_error_map = (squeeze(videoA(:,:,frame,:))-warped).^2;
            t_error_map = sum(t_error_map,3);
            t_error_map = sqrt(t_error_map);

            error_map(i,:) = amp_factor*t_error_map(:)';
            time = toc;
            fprintf('%fs\n',time);
        end
        clear t_error_map;

        fprintf('   - building cost matrix...');
        tic;
        cost_matrix = zeros(Kxform,Kxform,'int32');
        maxCost = 10;
        cost_matrix(:) = maxCost;
        for i = 1:Kxform
            cost_matrix(i,i) = 0;
        end
        cost_matrix = lambda*cost_matrix;
        cost_matrix = amp_factor * (cost_matrix/10);
        time = toc;
        fprintf('done. %fs\n',time);

        h = GCO_Create(height*width,Kxform);
        GCO_SetDataCost(h,error_map); % data term
        GCO_SetSmoothCost(h,cost_matrix);

        i_x_neighbor = sub2ind([height, width],...
                        y(:,1:end-1,:),...
                        x(:,1:end-1,:));
        j_x_neighbor = sub2ind([height, width],...
                        y(:,2:end,:),...
                        x(:,2:end,:));
        i_y_neighbor = sub2ind([height, width],...
                        y(1:end-1,:,:),...
                        x(1:end-1,:,:));
        j_y_neighbor = sub2ind([height, width],...
                        y(2:end,:,:),...
                        x(2:end,:,:));

        is = cat(1,i_x_neighbor(:), i_y_neighbor(:));
        js = cat(1,j_x_neighbor(:), j_y_neighbor(:));

        img = squeeze(videoA(:,:,frame,:));
        [m, n, ~] = size(img);
        [weights_x, weights_y] = weighted_derivative_ops_color(m, n, img, 3, .01, 0);
        weights_x = round(weights_x*10); weights_y = round(weights_y*10);
        weights = cat(1, weights_x, weights_y);

        S = sparse(is,js,weights,height*width, height*width);
        GCO_SetNeighbors(h,S);

        % optimization
        fprintf('   - GCO...');
        tic;
        GCO_Expansion(h); 
        uvw_label=GCO_GetLabeling(h);
        uvw_label=reshape(uvw_label,[height width]);
        time = toc;
        fprintf('done. %fs\n',time);

        u = zeros(height,width);
        v = zeros(height,width);
        w = zeros(height,width);
        for i=1:Kxform
            tmp_l = find(uvw_label==i);
            u(tmp_l) = uuu(i,tmp_l);
            v(tmp_l) = vvv(i,tmp_l);
            w(tmp_l) = www(i,tmp_l);
        end
        uvw_xform = cat(3,u,v,w);
        GCO_Delete(h);
    end % transform field
end % segmentUVW
