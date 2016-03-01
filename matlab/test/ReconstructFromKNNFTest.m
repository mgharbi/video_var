function tests = ReconstructFromKNNFTest
    tests = functiontests(localfunctions);
end % nlvvTest

function setupOnce(testCase)
    testCase.TestData.globals = init();
end

function testReconstruct(testCase)
    globals = testCase.TestData.globals;

    params = knnf_params();
    params.knn = 1;
    params.patch_size_time = 3;
    params.patch_size_space = 5;

    h = 10;
    w = 20;
    nF = 10;
    nC = 3;

    db  = uint8(255*rand(h,w,nF,nC));

    [X,Y,T] = meshgrid(0:w-1, 0:h-1, 0:nF-1);
    nnf = cat(4,X,Y,T);
    nnf(h-params.patch_size_space+2:h,:,:,:)  = 0;
    nnf(:,w-params.patch_size_space+2:w,:,:)   = 0;
    nnf(:,:,nF-params.patch_size_time+2:nF,:,:) = 0;
    nnf = int32(nnf);

    weights = single(ones(h,w,nF,1));

    [spatial,temporal] = viz_warp(single(nnf(:,:,:,1:3)));
    save_video(spatial,fullfile(globals.path.test_output, 'testReconstruct_spatial'), true);
    save_video(temporal,fullfile(globals.path.test_output, 'testReconstruct_temporal'), true);

    dx = nnf(:,:,:,1);
    dy = nnf(:,:,:,2);
    dt = nnf(:,:,:,3);
    assert(max(dx(:))<w-params.patch_size_space+1);
    assert(max(dy(:))<h-params.patch_size_space+1);
    assert(max(dt(:))<nF-params.patch_size_time+1);

    recons = reconstruct_from_knnf(db,nnf,weights, params);

    err = abs(single(recons)-single(db));
    err = max(err(:));
    err
    save_video(db,fullfile(globals.path.test_output, 'testReconstruct_input'), true);
    save_video(recons,fullfile(globals.path.test_output, 'testReconstruct_reconstruct'), true);
    assert(err < 1e-5);

end
