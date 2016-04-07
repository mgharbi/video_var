function tests = ConsistentVideoResizeTest
    tests = functiontests(localfunctions);
end % nlvvTest

function setupOnce(testCase)
    testCase.TestData.globals = init();
end

function testConsistentVideoResize(testCase)
    globals = testCase.TestData.globals;

    video = load_video(fullfile(globals.path.test_data,'rockettes01_video'));
    [h,w,nF,nC] = size(video);
    finest_res = [h,w,nF];
    src_scale = [1.0, 1.0, 1.0];
    dst_scale = [0.25, 0.25, 0.25];
    % dst_scale = [2.0, 2.0, 2.0];
    video_resized = consistent_video_resize(video, src_scale, dst_scale, finest_res);

    size(video)
    size(video_resized)

    save_video(video_resized,fullfile(globals.path.test_output, 'testConsistentVideoResize'), true);
end
