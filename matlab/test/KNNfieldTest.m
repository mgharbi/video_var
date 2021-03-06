function tests = KNNfieldTest
    tests = functiontests(localfunctions);
end % nlvvTest

function setupOnce(testCase)
    testCase.TestData.globals = init();
end

function testComputeKNNF(testCase)
    globals = testCase.TestData.globals;

    video  = uint8(255*rand(10,20,30,3));
    db = video;

    params = knnf_params();
    params.knn                    = 3;
    nnf                           = knnfield(video, db, params);

    % Check dimensions
    assert(size(nnf,1) == size(video,1));
    assert(size(nnf,2) == size(video,2));
    assert(size(nnf,3) == size(video,3));
    assert(size(nnf,4) == 3*params.knn);
    
    x = nnf(:,:,:,1);
    y = nnf(:,:,:,2);
    t = nnf(:,:,:,3);

    % Check range
    assert(max(x(:)) <= size(db,2)-params.patch_size_space);
    assert(max(y(:)) <= size(db,1)-params.patch_size_space);
    assert(max(t(:)) <= size(db,3)-params.patch_size_time);

    % Check that the NN field is optimal
    [X,Y,T] = meshgrid(1:size(video,2), 1:size(video,1), 1:size(video,3));
    X = X-1;
    X(end-params.patch_size_space+2:end, :, :) = 0;
    X(:, end-params.patch_size_space+2:end, :) = 0;
    X(:,:, end-params.patch_size_time+2:end) = 0;
    dx = abs(single(x) - X);
    assert(max(dx(:)) == 0)

    Y = Y-1;
    Y(end-params.patch_size_space+2:end, :, :) = 0;
    Y(:, end-params.patch_size_space+2:end, :) = 0;
    Y(:,:, end-params.patch_size_time+2:end) = 0;
    dy = abs(single(y) - Y);
    assert(max(dy(:)) == 0)

    T = T-1;
    T(end-params.patch_size_space+2:end, :, :) = 0;
    T(:, end-params.patch_size_space+2:end, :) = 0;
    T(:,:, end-params.patch_size_time+2:end) = 0;
    dt = abs(single(t) - T);
    assert(max(dt(:)) == 0)

    % Visualize
    [spatial,temporal] = viz_warp(single(nnf(:,:,:,1:3)));
    save_video(spatial,fullfile(globals.path.test_output, 'testComputeKNNF_spatial'), true);
    save_video(temporal,fullfile(globals.path.test_output, 'testComputeKNNF_temporal'), true);
end

function test2DNNF(testCase)
    globals = testCase.TestData.globals;
    video = imread(fullfile(globals.path.test_data,'rockettes01.png'));
    imwrite(video, fullfile(globals.path.test_output, 'test2DNNF_ref.png'));
    db = imread(fullfile(globals.path.test_data,'rockettes02.png'));
    imwrite(db, fullfile(globals.path.test_output, 'test2DNNF_db.png'));

    [h,w,c] = size(video);

    video = reshape(video, h,w,1,c);
    db = reshape(db, h,w,1,c);

    params = knnf_params();
    params.propagation_iterations = 2;
    params.patch_size_space       = 5;
    params.patch_size_time        = 1;
    params.knn                    = 5;

    tic;
    nnf                           = knnfield(video, db, params);
    toc

    for i = 1:params.knn
        warped = nnf_warp(db, size(video),nnf(:,:,:,3*(i-1)+1:3*i));
        imwrite(warped, fullfile(globals.path.test_output, sprintf('test2DNNF_warped_%d.png', i)));
    end

    [spatial,temporal] = viz_warp(single(nnf(:,:,:,1:3)));
    save_video(spatial,fullfile(globals.path.test_output, 'test2DNNF_spatial'), true);
    save_video(temporal,fullfile(globals.path.test_output, 'test2DNNF_temporal'), true);
end

function test3DNNF(testCase)
    globals = testCase.TestData.globals;

    video = load_video(fullfile(globals.path.test_data,'rockettes01_video'));
    db    = load_video(fullfile(globals.path.test_data,'rockettes02_video'));

    [h,w,c,nF] = size(video);

    params = knnf_params();
    params.propagation_iterations = 3;
    params.patch_size_space       = 5;
    params.patch_size_time        = 5;
    params.knn                    = 5;
    params.threads                = 8;

    tic;
    nnf = knnfield(video, db, params);
    toc

    for i = 1:params.knn
        warped = nnf_warp(db, size(video),nnf(:,:,:,3*(i-1)+1:3*i));
        save_video(warped, fullfile(globals.path.test_output, sprintf('test3DNNF_warped_%d', i)), true);
    end

    [spatial,temporal] = viz_warp(single(nnf(:,:,:,1:3)));
    save_video(spatial,fullfile(globals.path.test_output, 'test3DNNF_spatial'), true);
    save_video(temporal,fullfile(globals.path.test_output, 'test3DNNF_temporal'), true);
end

function testUniqueNeighbors(testCase)
    globals = testCase.TestData.globals;

    rng(0);
    video  = uint8(255*rand(10,20,30,3));
    db = video;

    params = knnf_params();
    params.propagation_iterations = 1;
    params.patch_size_space       = 5;
    params.patch_size_time        = 5;
    params.knn                    = 20;
    nnf                           = knnfield(video, db, params);
    
    [nnf,dist] = knnfield(video, db, params);

    [h,w,nF,knn] = size(nnf);
    knn = knn/3;

    unik = true(h,w,nF);
    for k = 1:knn
        eq = false(h,w,nF); % is one pair of channel equal ? (for each voxel)
        for j = (k+1):knn
            eq_x =  (nnf(:,:,:,3*(k-1)+1) == nnf(:,:,:,3*(j-1)+1));
            eq_y =  (nnf(:,:,:,3*(k-1)+2) == nnf(:,:,:,3*(j-1)+2));
            eq_t =  (nnf(:,:,:,3*(k-1)+3) == nnf(:,:,:,3*(j-1)+3));
            eq = eq | (eq_x & eq_y & eq_t); % Are all three coords equal?
        end
        unik = unik & ~eq;
    end
    unik(end-params.patch_size_space+1:end,:,:,:) = true;
    unik(:, end-params.patch_size_space+1:end,:,:) = true;
    unik(:,:,end-params.patch_size_time+1:end,:) = true;

    % Less than 1% of duplicate NN
    notu = find(~unik(:));
    assert(numel(notu)*1.0/numel(unik) < .01);
end
