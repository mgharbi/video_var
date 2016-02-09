#include <boost/filesystem.hpp>
#include <boost/test/unit_test.hpp>
#include <iostream>
namespace fs=boost::filesystem;

using namespace std;

struct DataFixture {
    fs::path outputPath;
    fs::path inputPath;
    fs::path paramPath;
    bool valid;
    DataFixture() {
        valid           = true;

        // basic path setup
        char **c        = boost::unit_test::framework::master_test_suite().argv;
        fs::path binDir = fs::system_complete(fs::path(c[0]).parent_path());
        outputPath      = fs::path(binDir/"../output/testResults/").normalize().parent_path();
        inputPath       = fs::path(binDir/"../data/testFixtures/").normalize().parent_path();
        paramPath       = fs::path(binDir/"../config/test.ini").normalize();


        fs::create_directory(outputPath);
        if(!fs::exists(outputPath)){
            BOOST_MESSAGE("Output path is not valid.");
            valid = false;
        }
        if(!fs::exists(inputPath)){
            BOOST_MESSAGE("Input path is not valid.");
            valid = false;
        }
        if(!fs::exists(paramPath)){
            BOOST_MESSAGE("Params path is not valid.");
            valid = false;
        }
    }
    ~DataFixture(){
        // fs::remove_all(outputPath);
    }
};
