#define BOOST_TEST_MODULE STWarpTests
#include <boost/test/unit_test.hpp>

#include "STWarp.hpp"
#include "WarpingField.hpp"
#include "Video.hpp"
#include "fixtures.hpp"
#include <vector>

BOOST_AUTO_TEST_SUITE( Initialization )
    BOOST_AUTO_TEST_CASE( ConfigFile ) {
        DataFixture f;
        BOOST_REQUIRE_MESSAGE(f.valid, "Path initialization failed.");

        FSTWarp warper = FSTWarp();
        fs::path paramsPath = f.inputPath/"testParams.ini";
        warper.loadParams(paramsPath);
        STWarpParams p = warper.getParams();
        BOOST_CHECK(!p.autoLevels);
        BOOST_CHECK_EQUAL(p.minPyrSize, 12);
        BOOST_CHECK_EQUAL(p.pyrSpacing, 2);
        BOOST_CHECK_EQUAL(p.warpIterations, 5);
        BOOST_CHECK_EQUAL(p.solverIterations, 10);
        BOOST_CHECK_EQUAL(p.outputPath.c_str(), "../bob");
    }

    BOOST_AUTO_TEST_CASE( WeightedMedian ) {
        int sz = 3;
        vector<int> vals(sz);
        vector<double> weights(sz);

        for (int i = 0; i < sz; ++i) {
            vals[i] = i+1;
            weights[i] = 1/( (double) sz);
        }

        int med = weightedMedian(vals, weights);
        BOOST_CHECK_EQUAL(med,2);
    }

    // BOOST_AUTO_TEST_CASE( Laplacian ) {
    //     DataFixture f;
    //     BOOST_REQUIRE_MESSAGE(f.valid, "Path initialization failed.");
    //
    //     int h = 40, w= 20, nF=30, nC=3;
    //     FVideo input(h,w,nF,nC);
    //     input.setAt(1,10,10,10,0);
    //
    //     FVideo weight(input.size());
    //     weight.reset(1);
    //
    //     FVideo output(input.size());
    //     FSTWarp warper = FSTWarp();
    //     STWarpParams p;
    //     for (int i = 0; i < 4; ++i) {
    //         p.lambda[i] = 1;
    //     }
    //     warper.setParams(p);
    //     warper.weightedLaplacian(input,weight,output);
    //     
    //     BOOST_CHECK_EQUAL(output.at(10,10,10,1),0);
    //     BOOST_CHECK_EQUAL(output.at(10,10,10,2),0);
    //
    //     BOOST_CHECK_EQUAL(output.at(10,10,10,0),-6);
    //     BOOST_CHECK_EQUAL(output.at(11,10,10,0),1);
    //     BOOST_CHECK_EQUAL(output.at(9,10,10,0),1);
    //     BOOST_CHECK_EQUAL(output.at(10,11,10,0),1);
    //     BOOST_CHECK_EQUAL(output.at(10,9,10,0),1);
    //     BOOST_CHECK_EQUAL(output.at(10,10,11,0),1);
    //     BOOST_CHECK_EQUAL(output.at(10,10,9,0),1);
    //     BOOST_CHECK_EQUAL(output.at(10,10,12,0),0);
    //
    //     input.setAt(1,11,10,10,0);
    //     input.setAt(1,10,11,10,0);
    //     input.setAt(1,10,10,11,0);
    //     output.reset();
    //     warper.weightedLaplacian(input,weight,output);
    //     BOOST_CHECK_EQUAL(output.at(10,10,10,0),-3);
    //     BOOST_CHECK_EQUAL(output.at(11,10,10,0),-5);
    //     BOOST_CHECK_EQUAL(output.at(10,11,10,0),-5);
    //     BOOST_CHECK_EQUAL(output.at(10,10,11,0),-5);
    //     BOOST_CHECK_EQUAL(output.at(12,10,10,0),1);
    //     BOOST_CHECK_EQUAL(output.at(13,10,10,0),0);
    //
    //     weight.setAt(-1,11,10,10,0);
    //     weight.setAt(2,10,11,10,0);
    //     weight.setAt(3,10,10,11,0);
    //     output.reset();
    //     warper.weightedLaplacian(input,weight,output);
    //     // BOOST_CHECK_EQUAL(output.at(10,10,10,0),-2);
    //     // BOOST_CHECK_EQUAL(output.at(11,10,10,0),7);
    //     // BOOST_CHECK_EQUAL(output.at(10,11,10,0),-5);
    //     // BOOST_CHECK_EQUAL(output.at(10,10,11,0),-5);
    //     // BOOST_CHECK_EQUAL(output.at(12,10,10,0),1);
    //     // BOOST_CHECK_EQUAL(output.at(13,10,10,0),0);
    // }

    BOOST_AUTO_TEST_CASE( TestRun ) {
        DataFixture f;
        BOOST_REQUIRE_MESSAGE(f.valid, "Path initialization failed.");

        fs::path videoPathA = f.inputPath/"videoA.mov";
        fs::path videoPathB = f.inputPath/"videoB.mov";
        STWarpParams params(f.paramPath);
        params.name = "testrun";
        params.outputPath = f.outputPath;

        FSTWarp warper = FSTWarp();
        warper.setParams(params);
        warper.loadVideos(videoPathA, videoPathB);

        FWarpingField uvw = warper.computeWarp();
        uvw.exportSpacetimeMap(f.outputPath,"testSTWarp");

        IVideo backward(videoPathB);
        VideoProcessing::backwardWarp(backward,uvw);
        IVideo fusedB(uvw.getHeight(),uvw.getWidth(),uvw.frameCount(),3);
        VideoProcessing::videoFuse(backward,IVideo(videoPathA),fusedB);
        fusedB.exportVideo(params.outputPath,(params.name+"_fused_backward"));

        BOOST_CHECK(fs::exists(f.outputPath/"testSTWarp-space.mov"));
        BOOST_CHECK(fs::exists(f.outputPath/"testSTWarp-time.mov"));
    }
    // BOOST_AUTO_TEST_CASE( TestRun ) {
    //     DataFixture f;
    //     BOOST_REQUIRE_MESSAGE(f.valid, "Path initialization failed.");

    //     fs::path videoPathA = f.inputPath/"videoA_large.mov";
    //     fs::path videoPathB = f.inputPath/"videoB_large.mov";

    //     FSTWarp warper = FSTWarp();
    //     warper.loadVideos(videoPathA, videoPathB);
    //     FWarpingField uvw = warper.computeWarp();
    //     uvw.exportSpacetimeMap(f.outputPath,"testSTWarp");
    //     BOOST_CHECK(fs::exists(f.outputPath/"testSTWarp-space.mov"));
    //     BOOST_CHECK(fs::exists(f.outputPath/"testSTWarp-time.mov"));
    // }
    BOOST_AUTO_TEST_CASE( BinaryInOut ) {
        DataFixture f;
        BOOST_REQUIRE_MESSAGE(f.valid, "Path initialization failed.");

        FWarpingField warpField(100,80,60,3);
        warpField.setAt(-10, 20,30,15,2);
        warpField.setAt(10, 30,10,50,1);

        fs::path outputFile = f.outputPath/"warpField.stw";
        warpField.save(outputFile);
        BOOST_REQUIRE(fs::exists(outputFile));

        FWarpingField warpField2;
        warpField2.load(outputFile);
        BOOST_CHECK_EQUAL(warpField.getHeight(),warpField2.getHeight());
        BOOST_CHECK_EQUAL(warpField.getWidth(),warpField2.getWidth());
        BOOST_CHECK_EQUAL(warpField.elementCount(),warpField2.elementCount());
        BOOST_CHECK_EQUAL(warpField.frameCount(),warpField2.frameCount());
        BOOST_CHECK_EQUAL(warpField.channelCount(),warpField2.channelCount());
        BOOST_CHECK_EQUAL(warpField.at(20,30,15,2),warpField2.at(20,30,15,2));
        BOOST_CHECK_EQUAL(warpField.at(30,10,50,1),warpField2.at(30,10,50,1));
    }
    BOOST_AUTO_TEST_CASE( ResampleWarpField ) {
        int h = 100, w = 200, nF = 30, nC = 3;
        FWarpingField warpField = FWarpingField(h,w,nF,nC);

        FSTWarp warper = FSTWarp();

        vector<int> dims(3);
        dims[0] = h/2;
        dims[1] = w/2;
        dims[2] = nF/2;
        warper.resampleWarpingField(warpField,dims);

        BOOST_CHECK(warpField.getHeight() == h/2);
        BOOST_CHECK(warpField.getWidth() == w/2);
        BOOST_CHECK(warpField.frameCount() == nF/2);
        BOOST_CHECK(warpField.channelCount() == nC);
    }
BOOST_AUTO_TEST_SUITE_END()
