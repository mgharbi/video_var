#define BOOST_TEST_MODULE VideoProcessingTests
#include <boost/test/unit_test.hpp>

#include "Video.hpp"
#include "VideoProcessing.hpp"
#include "fixtures.hpp"
#include <boost/filesystem.hpp>
#include <vector>

BOOST_AUTO_TEST_SUITE( Interpolation )
    BOOST_AUTO_TEST_CASE( dilatation3d ) {
        IVideo source(10,10,10,1);

        source.reset(0);

        unsigned char val = 255;
        source.setAt(val, 5,5,5,0 );

        int dims[3] = {0, 1, 2};
        VideoProcessing::dilate3D(source,dims);
        for (int k = -dims[2]; k <= dims[2]; ++k)
            for (int j = -dims[1]; j <= dims[1]; ++j)
                for (int i = -dims[0]; i <= dims[0]; ++i)
        {
            BOOST_CHECK_EQUAL(source.at(5+i,5+j,5+k,0),val);
        }
        
    }
    BOOST_AUTO_TEST_CASE( TrilinearInterpolation ) {
        DVideo source(4,4,4,1);

        source.reset(1);

        double pTarget[1];

        double queryPoint[3];
        for (int i = 0; i < 3; ++i) {
            queryPoint[i] = .5;
        }
        VideoProcessing::trilinearInterpolate(source, queryPoint, pTarget);
        BOOST_CHECK_EQUAL(pTarget[0], 1);

        for (int j = 0; j < 2; ++j)
            for (int k = 0; k < 2; ++k)
        {
            source.setAt(2,1,j,k,0);
        }

        VideoProcessing::trilinearInterpolate(source, queryPoint, pTarget);
        BOOST_CHECK_EQUAL(pTarget[0], 1.5);
    }

    BOOST_AUTO_TEST_CASE( Resize ) {
        int h = 10, w = 20, nF = 2, nC = 3;
        IVideo v(h, w, nF, nC);
        unsigned char *data = v.channelWriter(0);
        vector<int> dims = v.dimensions();

        for (int k = 0; k < nF; ++k) 
            for (int j = 0; j < w; ++j) 
                for (int i = 0; i < h; ++i) 
        {
            if( i%2 == 0 ){
                data[i+h*(j+w*(k+2*nF))] = 255;
            }else{
                data[i+h*(j+w*(k+nF))] = 255;
            }
        }

        IVideo v2(2*h,2*w,2*nF,nC);
        IVideo v3(.5*h,.5*w,nF,nC);
        VideoProcessing::resize(v,&v2);
        VideoProcessing::resize(v,&v3);
        
        BOOST_CHECK_EQUAL(v2.getHeight(), 2*h);
        BOOST_CHECK_EQUAL(v2.getWidth(), 2*w);
        BOOST_CHECK_EQUAL(v2.frameCount(), 2*nF);
        BOOST_CHECK_EQUAL(v2.channelCount(), nC);
        BOOST_CHECK_EQUAL(v2.at(0,0,0,1), v.at(0,0,0,1));
        BOOST_CHECK_EQUAL(v2.at(0,0,0,2), v.at(0,0,0,2));
        BOOST_CHECK_EQUAL(v2.at(1,0,0,1), 127);
        BOOST_CHECK_EQUAL(v2.at(1,0,0,2), 127);
        BOOST_CHECK_EQUAL(v2.at(2,0,0,1), v.at(1,0,0,1));
        BOOST_CHECK_EQUAL(v2.at(2,0,0,2), v.at(1,0,0,2));

        BOOST_CHECK_EQUAL(v3.getHeight(),(int) (.5*h));
        BOOST_CHECK_EQUAL(v3.getWidth(),(int)(.5*w));
        BOOST_CHECK_EQUAL(v3.frameCount(), nF);
        BOOST_CHECK_EQUAL(v3.channelCount(), nC);
        BOOST_CHECK_EQUAL(v3.at(0,0,0,1), v.at(0,0,0,1));
        BOOST_CHECK_EQUAL(v3.at(0,0,0,2), v.at(0,0,0,2));
        BOOST_CHECK_EQUAL(v3.at(1,0,0,1), v.at(2,0,0,1));
        BOOST_CHECK_EQUAL(v3.at(1,0,0,2), v.at(2,0,0,2));
    }

    BOOST_AUTO_TEST_CASE( Filtering ) {
        int h = 100, w = 200, nF = 100, nC = 3;
        IVideo v(h, w, nF, nC);
        unsigned char *data = v.channelWriter(0);
        vector<int> dims = v.dimensions();

        IVideo v2 = IVideo(v.size());
        int filterSize = 1;
        double* filter = new double[2*filterSize+1];
        for (int i = 0; i < 2*filterSize + 1; ++i) {
            filter[i] = 1;
        }

        // Test Horizontal filter
        for (int k = 0; k < nF; ++k) 
            for (int j = 0; j < w; ++j) 
                for (int i = 0; i < h; ++i) 
        {
            if( j==20){
                data[i+h*(j+w*(k+2*nF))] = 255;
            }
            if( j==30 ){
                data[i+h*(j+w*(k+1*nF))] = 255;
            }
            if( j==40 ){
                data[i+h*(j+w*(k+0*nF))] = 255;
            }
        }
        VideoProcessing::hfiltering(v,v2,filter,filterSize);
        BOOST_CHECK(v2.at(0,39,0,0)==255);
        BOOST_CHECK(v2.at(0,40,0,0)==255);
        BOOST_CHECK(v2.at(0,41,0,0)==255);
        BOOST_CHECK(v2.at(0,29,0,1)==255);
        BOOST_CHECK(v2.at(0,30,0,1)==255);
        BOOST_CHECK(v2.at(0,31,0,1)==255);
        BOOST_CHECK(v2.at(0,19,0,2)==255);
        BOOST_CHECK(v2.at(0,20,0,2)==255);
        BOOST_CHECK(v2.at(0,21,0,2)==255);

        // Test Vertical filter
        v.reset();
        for (int k = 0; k < nF; ++k) 
            for (int j = 0; j < w; ++j) 
                for (int i = 0; i < h; ++i) 
        {
            if( i==20){
                data[i+h*(j+w*(k+2*nF))] = 255;
            }
            if( i==30 ){
                data[i+h*(j+w*(k+1*nF))] = 255;
            }
            if( i==40 ){
                data[i+h*(j+w*(k+0*nF))] = 255;
            }
        }
        VideoProcessing::vfiltering(v,v2,filter,filterSize);
        BOOST_CHECK(v2.at(39,0,0,0)==255);
        BOOST_CHECK(v2.at(40,0,0,0)==255);
        BOOST_CHECK(v2.at(41,0,0,0)==255);
        BOOST_CHECK(v2.at(29,0,0,1)==255);
        BOOST_CHECK(v2.at(30,0,0,1)==255);
        BOOST_CHECK(v2.at(31,0,0,1)==255);
        BOOST_CHECK(v2.at(19,0,0,2)==255);
        BOOST_CHECK(v2.at(20,0,0,2)==255);
        BOOST_CHECK(v2.at(21,0,0,2)==255);

        // Test Time filter
        v.reset();
        for (int k = 0; k < nF; ++k) 
            for (int j = 0; j < w; ++j) 
                for (int i = 0; i < h; ++i) 
        {
            if( k==20){
                data[i+h*(j+w*(k+2*nF))] = 255;
            }
            if( k==30 ){
                data[i+h*(j+w*(k+1*nF))] = 255;
            }
            if( k==40 ){
                data[i+h*(j+w*(k+0*nF))] = 255;
            }
        }
        VideoProcessing::tfiltering(v,v2,filter,filterSize);
        BOOST_CHECK(v2.at(0,0,39,0)==255);
        BOOST_CHECK(v2.at(0,0,40,0)==255);
        BOOST_CHECK(v2.at(0,0,41,0)==255);
        BOOST_CHECK(v2.at(0,0,29,1)==255);
        BOOST_CHECK(v2.at(0,0,30,1)==255);
        BOOST_CHECK(v2.at(0,0,31,1)==255);
        BOOST_CHECK(v2.at(0,0,19,2)==255);
        BOOST_CHECK(v2.at(0,0,20,2)==255);
        BOOST_CHECK(v2.at(0,0,21,2)==255);

        delete filter;

        // TODO: test out of bound
    }
    BOOST_AUTO_TEST_CASE( GaussianFiltering ) {
        DataFixture f;
        int h = 100, w = 200, nF = 40, nC = 1;
        DVideo v(h, w, nF, nC);
        vector<int> dims = v.dimensions();

        int sz = 10;
        for(int j = -sz; j < sz; ++j)
            for(int i = -sz; i < sz; ++i)
        {
            v.setAt(255,30+i,20+j,20,0);
        }

        double sigma[3] = {4,4,0};
        int fsize[3] = {8,8,0};
        VideoProcessing::gaussianSmoothing(v, sigma, fsize);

        for(int j = -sz-fsize[1]; j < sz+fsize[1]; ++j)
            for(int i = -sz-fsize[0]; i < sz+fsize[0]; ++i)
        {
            BOOST_CHECK(v.at(30+i,20+j,20,0)>0);
        }
    }

    BOOST_AUTO_TEST_CASE( Derivatives ) {
        IVideo v(20,20,20,1);
        DVideo dx(v.size());
        DVideo dy(v.size());
        DVideo dt(v.size());

        int center[3] = {10,10,10};
        int sz = 2;
        for (int k = -sz; k < sz; ++k)
            for (int j = -sz; j < sz; ++j)
                for (int i = -sz; i < sz; ++i)
        {
            v.setAt(255, center[0]+i,center[1]+j,center[2]+k,0);
        }

        VideoProcessing::dx(v,dx,false);
        VideoProcessing::dy(v,dy,false);
        VideoProcessing::dt(v,dt,false);

        BOOST_CHECK_EQUAL(dx.at(center[0],center[1]-sz-1,center[2],0), 255);
        BOOST_CHECK_EQUAL(dx.at(center[0],center[1]+sz-1,center[2],0), -255);
        BOOST_CHECK_EQUAL(dy.at(center[0]-sz-1,center[1],center[2],0), 255);
        BOOST_CHECK_EQUAL(dy.at(center[0]+sz-1,center[1],center[2],0), -255);
        BOOST_CHECK_EQUAL(dt.at(center[0],center[1],center[2]-sz-1,0), 255);
        BOOST_CHECK_EQUAL(dt.at(center[0],center[1],center[2]+sz-1,0), -255);
    }

    BOOST_AUTO_TEST_CASE( BackwarpWarp ) {
        DataFixture f;
        fs::path path = f.inputPath/"videoA.mov";
        IVideo v(path);
        IVideo warped(v.size());
        DVideo warpField(v.size());

        vector<int> dims = v.dimensions();
        double* pWarp = warpField.dataWriter();
        int nVoxels = v.voxelCount();

        // Construct a test warpField
        int index;
        for (int k = 0; k < dims[2]; ++k)
            for (int j = 0; j < dims[1]; ++j)
                for (int i = 0; i < dims[0]; ++i)
        {
            index = i+dims[0]*(j+dims[1]*k);
            pWarp[index] = 10;
            pWarp[index+nVoxels] = 10;
        }

        VideoProcessing::backwardWarp(v,warpField,warped);

        BOOST_CHECK_EQUAL(warped.at(dims[0]-1,dims[1]-1,1,0),0);
        BOOST_CHECK_EQUAL(warped.at(0,0,0,0),v.at(10,10,0,0));
    }
    BOOST_AUTO_TEST_CASE( Edges3D ) {
        DataFixture f;
        fs::path path = f.inputPath/"derailleur_01.mov";
        IVideo v(path);

        IVideo edges;

        VideoProcessing::edges3D(v,edges);

        // int dilateSz[3] = {2,2,0};
        // VideoProcessing::dilate3D(edges, dilateSz);

        edges.exportVideo(f.outputPath,"edgeMap3D");

    }
    BOOST_AUTO_TEST_CASE( ForwardWarp ) {
        DataFixture f;
    }
BOOST_AUTO_TEST_SUITE_END()
 
