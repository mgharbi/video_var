#define BOOST_TEST_MODULE VideoTests
#include <boost/test/unit_test.hpp>

#include "Video.hpp"
#include <boost/filesystem.hpp>
#include "fixtures.hpp"

BOOST_AUTO_TEST_SUITE( VideoOutput )
    // TODO: templated test cases.
    BOOST_AUTO_TEST_CASE( BasicConstructors ) {

        IVideo vi;
        BOOST_CHECK(vi.elementCount() == 0);

        DVideo vd;
        BOOST_CHECK(vd.elementCount() == 0);

        IVideo v2(10,20,30,3);
        BOOST_CHECK(v2.elementCount() == 10*20*30*3);
        BOOST_CHECK(v2.voxelCount() == 10*20*30);
        BOOST_CHECK(v2.getHeight() == 10);
        BOOST_CHECK(v2.getWidth() == 20);
        BOOST_CHECK(v2.frameCount() == 30);
        BOOST_CHECK(v2.channelCount() == 3);

        IVideo v3(VideoSize(11, 21, 31, 2));
        BOOST_CHECK(v3.elementCount() == 11*21*31*2);

        vector<int> dims = v3.dimensions();
        BOOST_CHECK(dims[0] == 11);
        BOOST_CHECK(dims[1] == 21);
        BOOST_CHECK(dims[2] == 31);
        BOOST_CHECK(dims[3] == 2);
        VideoSize sz = v3.size();
        BOOST_CHECK(sz.height == 11);
        BOOST_CHECK(sz.width == 21);
        BOOST_CHECK(sz.nFrames == 31);
        BOOST_CHECK(sz.nChannels == 2);
        const unsigned char* pDataReader = v3.dataReader();
        const unsigned char* pChannelReader = v3.channelReader(1);
        v3.setAt(10,9,15,15,1);
        BOOST_CHECK(pDataReader[9+dims[0]*(15+dims[1]*(15 + dims[2]))] == 10);
        BOOST_CHECK(pChannelReader[9+dims[0]*(15+dims[1]*(15))] == 10);
        BOOST_CHECK(v3.at(9, 15, 15, 1) == 10);
        BOOST_CHECK(v3.max() == 10);
        IVideo channel = v3.extractChannel(1);
        BOOST_CHECK(channel.at(9,15,15,0) == 10);

        unsigned char *pDataWriter = v3.dataWriter();
        unsigned char *pChannelWriter = v3.channelWriter(0);
        pDataWriter[13] = 19;
        BOOST_CHECK(pChannelWriter[13] == 19);
        pChannelWriter[13] = 21;

        v3.reset(8);
        BOOST_CHECK(v3.min() == 8);
        v3.scalarMultiply(2);
        BOOST_CHECK(v3.at(1,1,1,1) == 16);
        v3.scalarMultiplyChannel(2,0);
        BOOST_CHECK(v3.at(1,1,1,0) == 32);
        BOOST_CHECK(v3.at(1,1,1,1) == 16);

        v3.reset(10);
        IVideo v4(v3);
        IVideo v5 = v3;
        BOOST_CHECK(pDataReader != v4.dataReader());
        BOOST_CHECK(pDataReader != v5.dataReader());
        const unsigned char *p4 = v4.dataReader();
        const unsigned char *p5 = v5.dataReader();
        for (int i = 0; i < v3.elementCount(); ++i) {
            BOOST_CHECK(p4[i] == pDataReader[i]);
            BOOST_CHECK(p5[i] == pDataReader[i]);
        }

        v3.add(v4);
        BOOST_CHECK(v3.at(1,2,3,1) == 20);
        v3.subtract(v5);
        BOOST_CHECK(v3.at(1,2,3,1) == 10);

        int h = 10, w = 20, nF = 30, nC = 3;
        IVideo v(h, w, nF, nC);

        BOOST_REQUIRE( v.voxelCount() == h*w*nF);
        BOOST_REQUIRE( v.elementCount() == h*w*nF*nC);

        BOOST_REQUIRE( v.dataReader() != NULL );
        BOOST_REQUIRE( v.dataWriter() != NULL );
        BOOST_REQUIRE( v.channelReader(0) != NULL );
        BOOST_REQUIRE( v.channelReader(1) != NULL );
        BOOST_REQUIRE( v.channelReader(2) != NULL );
        BOOST_REQUIRE( v.channelWriter(0) != NULL );
        BOOST_REQUIRE( v.channelWriter(1) != NULL );
        BOOST_REQUIRE( v.channelWriter(2) != NULL );

        IVideo s1(5,5,5,1);
        IVideo s2(6,5,5,1);
        BOOST_CHECK_THROW(s1.multiply(s2), IncorrectSizeException);
    }

    BOOST_AUTO_TEST_CASE( InputOutput ) {
        DataFixture f;
        BOOST_REQUIRE_MESSAGE(f.valid, "Path initialization failed.");

        IVideo v(f.inputPath/"videoA.mov");
        BOOST_CHECK_EQUAL(v.frameCount(), 34);
        BOOST_CHECK_EQUAL(v.getHeight(), 50);
        BOOST_CHECK_EQUAL(v.getWidth(), 90);
        BOOST_CHECK_EQUAL(v.channelCount(), 3);

        fs::path outputFolder = f.outputPath/"testVideo";
        fs::create_directory(outputFolder);
        BOOST_REQUIRE_MESSAGE(fs::exists(outputFolder), "Output folder could not be created.");
        fs::path outputImage = outputFolder/"frame1.jpg";
        v.exportFrame(1,outputImage);
        BOOST_REQUIRE(fs::exists(outputImage));

        string videoName = "testInstantiateOutput";
        v.exportVideo(outputFolder, videoName);
        videoName += ".mov";
        BOOST_REQUIRE(fs::exists(outputFolder/videoName));
        IVideo v2(outputFolder/videoName);
        BOOST_CHECK_EQUAL(v2.getHeight(), v.getHeight());
        BOOST_CHECK_EQUAL(v2.getWidth(), v.getWidth());
        BOOST_CHECK_EQUAL(v2.frameCount(), v.frameCount());
        BOOST_CHECK_EQUAL(v2.channelCount(), v.channelCount());
    }
BOOST_AUTO_TEST_SUITE_END()
