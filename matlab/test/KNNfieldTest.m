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

    params                        = struct();
    params.propagation_iterations = 3;
    params.patch_size_space       = 5;
    params.patch_size_time        = 3;
    params.knn                    = 3;
    nnf                           = knnfield(video, db, params);

    size(nnf)

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

    % % Check that the NN field is optimal
    % [X,Y,T] = meshgrid(1:size(video,2), 1:size(video,1), 1:size(video,3));
    % X = X-1;
    % X(end-params.patch_size_space+2:end, :, :) = 0;
    % X(:, end-params.patch_size_space+2:end, :) = 0;
    % X(:,:, end-params.patch_size_time+2:end) = 0;
    % dx = abs(single(x) - X);
    % assert(max(dx(:)) == 0)
    %
    % Y = Y-1;
    % Y(end-params.patch_size_space+2:end, :, :) = 0;
    % Y(:, end-params.patch_size_space+2:end, :) = 0;
    % Y(:,:, end-params.patch_size_time+2:end) = 0;
    % dy = abs(single(y) - Y);
    % assert(max(dy(:)) == 0)
    %
    % T = T-1;
    % T(end-params.patch_size_space+2:end, :, :) = 0;
    % T(:, end-params.patch_size_space+2:end, :) = 0;
    % T(:,:, end-params.patch_size_time+2:end) = 0;
    % dt = abs(single(t) - T);
    % assert(max(dt(:)) == 0)

    % Visualize
    % [spatial,temporal] = viz_warp(single(nnf(:,:,:,1:3)));
    % save_video(spatial,fullfile(globals.path.test_output, 'testComputeKNNF_spatial'), true);
    % save_video(temporal,fullfile(globals.path.test_output, 'testComputeKNNF_temporal'), true);

end
