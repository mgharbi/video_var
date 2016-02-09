#include <iostream>
#include "STWarp.hpp" 
#include "Renderer.hpp"
#include "WarpingField.hpp"
#include "Video.hpp"
#include "VideoProcessing.hpp"
#include <boost/filesystem.hpp>
#include <boost/date_time.hpp>
#include <cassert>


using namespace std;
namespace fs = boost::filesystem;
namespace bt = boost::gregorian;

typedef float precision_t;

int main(int argc, const char **argv) {
    fs::path binDir     = fs::system_complete(fs::path(argv[0]).parent_path());
    fs::path outputPath = fs::path(binDir/"../output/").normalize().parent_path();
    fs::path inputPath  = fs::path(binDir/"../data/").normalize().parent_path();

    // Create date folder
    bt::date today = bt::day_clock::local_day();
    const std::locale fmt(std::locale::classic(), new bt::date_facet("%Y-%m-%d"));
    std::ostringstream os;
    os.imbue(fmt);
    os << today;
    outputPath /= os.str();
    fs::create_directory(outputPath);

    string name;
    if(argc < 2) {
        name = "derailleur";
    }else {
        name = argv[1];
    }

    std::string ext        = ".mov";
    fs::path videoPathA    = inputPath/(name+"_01"+ext);
    fs::path videoPathB    = inputPath/(name+"_02"+ext);
    fs::path maskPathA     = inputPath/(name+"_01_mask"+ext);
    if(!fs::exists(videoPathA) || !fs::exists(videoPathB)){
        printf("Input file not found, aborting.\n");
        return 1;
    }
    fs::path paramPath = fs::path(binDir/"../config/default.ini").normalize();


    STWarpParams params(paramPath);
    params.name = name;
    params.outputPath = outputPath/params.name;
    fs::create_directories(params.outputPath);

    // Params dump
    if( fs::exists(paramPath)){
        fs::path paramDump = params.outputPath/(name+"_params.ini");
        if( fs::exists(paramDump)){
            fs::remove(paramDump);
        }
        fs::copy(paramPath, paramDump);
    }

    WarpingField<precision_t> uvw;
    fs::path warpFieldPath = params.outputPath/(name+".stw");

    STWarp<precision_t> warper = STWarp<precision_t>();
    warper.setParams(params);
    warper.loadVideos(videoPathA, videoPathB);
    // if(fs::exists(maskPathA)){
    //     warper.loadMasks(maskPathA, maskPathA);
    // }

    // Compute optical flow
    if(params.useFlow){
        fs::path flowApath = params.outputPath/(name+"_flowA.stw");
        // fs::path flowBpath = params.outputPath/(name+"-flowB.stw");
        if(fs::exists(flowApath)) {
            warper.flowA->load(flowApath);
            // warper.flowB->load(flowBpath);
        }else{
            warper.computeOpticalFlow();
            warper.flowA->save(flowApath);
            // warper.flowB->save(flowBpath);
        }

        warper.flowA->exportSpacetimeMap(params.outputPath,name+"-flowA");
        // warper.flowB->exportSpacetimeMap(params.outputPath,name+"-flowB");
    }

    if(fs::exists(warpFieldPath)) {
        uvw.load(warpFieldPath);
    }else{
        // fs::path xformPath = params.outputPath/(name+"_xform.stw");
        fs::path transPath = params.outputPath/(name+"_trans.stw");
        // if(fs::exists(xformPath) && fs::exists(transPath)) {
        //     WarpingField<precision_t> xform, trans, nnf;
        //     xform.load(xformPath);
        //     trans.load(transPath);
        //     nnf = WarpingField<precision_t>(xform.size());
        //     xform.exportSpacetimeMap(params.outputPath, (name + "_xform"));
        //     trans.exportSpacetimeMap(params.outputPath, (name + "_trans"));
        //     fprintf(stderr,"+ Fusing NNF fields...");
        //     nnf.exportSpacetimeMap(params.outputPath, (name + "_nnf"));
        //     warper.setInitialWarpField(nnf);
        //     fprintf(stderr, "done\n");
        // }
        if(fs::exists(transPath)) {
            fprintf(stderr, "Using initial warping field\n");
            WarpingField<precision_t> trans;
            trans.load(transPath);
            trans.exportSpacetimeMap(params.outputPath, (name + "_trans"));
            warper.setInitialWarpField(trans);
        }
        uvw = warper.computeWarp();
        uvw.save(warpFieldPath);
    }

    uvw.exportSpacetimeMap(params.outputPath, name);

    IVideo videoA(videoPathA);
    IVideo videoB(videoPathB);

    IVideo backward(videoA.size());
    VideoProcessing::backwardWarp(videoB,uvw,backward);
    IVideo fusedB(uvw.getHeight(),uvw.getWidth(),uvw.frameCount(),3);
    VideoProcessing::videoFuse(backward,videoA,fusedB);
    fusedB.exportVideo(params.outputPath,(name+"_backward"));
    backward.exportVideo(params.outputPath,(name+"_bw"));

    Video<precision_t> cost(videoA.size());
    cost.copy(videoA);
    cost.subtract(backward);
    cost.collapse();
    precision_t * pCost = cost.dataWriter();
    for (int i = 0; i < cost.voxelCount(); ++i) {
        pCost[i] = abs(pCost[i]);
    }
    cost.exportVideo(params.outputPath,(name+"_error"));

    WarpingField<precision_t> space(uvw);
    space.scalarMultiplyChannel(0,2);
    backward = IVideo(videoA.size());
    VideoProcessing::backwardWarp(videoB,uvw,backward);
    backward.exportVideo(params.outputPath,(name+"_bw_space"));

    WarpingField<precision_t> time(uvw);
    time.scalarMultiplyChannel(0,0);
    time.scalarMultiplyChannel(0,1);
    backward = IVideo(videoA.size());
    VideoProcessing::backwardWarp(videoB,time,backward);
    fusedB.reset();
    VideoProcessing::videoFuse(backward,videoA,fusedB);
    fusedB.exportVideo(params.outputPath,(name+"_bw_time"));

    IVideo fused;
    VideoProcessing::videoFuse(videoB,videoA,fused);
    fused.exportVideo(params.outputPath,(name+"_original"));

    fs::path flowApath = params.outputPath/(name+"_flowA.stw");
    // fs::path flowBpath = params.outputPath/(name+"-flowB.stw");
    WarpingField<precision_t> *flowA;
    if(fs::exists(flowApath)) {
        flowA = new WarpingField<precision_t>(flowApath);
    }else{
        flowA = NULL;
    }
        
    double exa[3] = {1,1,1};
    Renderer<precision_t> renderer(params);
    IVideo warped;
    warped = renderer.render(videoA,videoB,uvw,exa,flowA);
    warped.exportVideo(params.outputPath,(name+"_aligned"));

    exa[0] = -2; exa[1] = -2; exa[2] = 0;
    warped = renderer.render(videoA,videoB,uvw,exa,flowA);
    warped.exportVideo(params.outputPath,(name+"_space2"));

    exa[0] = -4; exa[1] = -4; exa[2] = 0;
    warped = renderer.render(videoA,videoB,uvw,exa,flowA);
    warped.exportVideo(params.outputPath,(name+"_space4"));


    if(flowA != NULL){
        delete flowA;
    }

    // return 0;
}
