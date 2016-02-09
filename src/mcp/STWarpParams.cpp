/* --------------------------------------------------------------------------
 * File:    STWarpParams.cpp
 * Author:  Michael Gharbi <gharbi@mit.edu>
 * Created: 2014-02-28
 * --------------------------------------------------------------------------
 * 
 * 
 * 
 * ------------------------------------------------------------------------*/

 
#include "mcp/STWarpParams.hpp"


/*
 * Load the parameters from a standard .ini file.
 */
void STWarpParams::loadParams(fs::path path) {
    po::options_description desc("Allowed options");
    desc.add_options()
        ("STWarp.autoLevels",po::value<bool>(),"automatic pyramid levels")
        ("STWarp.minPyrSize",po::value<int>(),"min dimension in the pyramid")
        ("STWarp.pyrSpacing",po::value<double>(),"scale factor between two pyramid levels")
        ("STWarp.warpIterations",po::value<int>(),"warping iterations per pyramid level")
        ("STWarp.solverIterations",po::value<int>(),"iterations of the linear solver")
        ("STWarp.output",po::value<std::string>(),"output directory")
        ("STWarp.lambdaSUV",po::value<double>(),"spatial regularization uv")
        ("STWarp.lambdaTUV",po::value<double>(),"temporal regularization uv")
        ("STWarp.lambdaSW",po::value<double>(),"spatial regularization w")
        ("STWarp.lambdaTW",po::value<double>(),"temporal regularization w")
        ("STWarp.lambdaMult",po::value<double>(),"global regularization multiplier")
        ("STWarp.localSmoothness",po::value<double>(),"local smoothness relative weight")
        ("STWarp.localSmoothnessPrior",po::value<double>(),"local smoothness sigma")
        ("STWarp.divPrior",po::value<double>(),"sigma for occlusion divergence prior")
        ("STWarp.occlusionPrior",po::value<double>(),"sigma for data fidelity divergence prior")
        ("STWarp.useOcclusion",po::value<bool>(),"use color information")
        ("STWarp.useColor",po::value<bool>(),"use color information")
        ("STWarp.useFeatures",po::value<bool>(),"use edge information")
        ("STWarp.bypassTimeWarp",po::value<bool>(),"dont compute the timewarp map")
        ("STWarp.decoupleRegularization",po::value<int>(),"decouple uv and w regularization")
        ("STWarp.debug",po::value<bool>(),"display debug info")
        ("STWarp.c",po::value<double>(),"speed scale c for coord (x,y,ct)")
        ("STWarp.medfiltSize",po::value<int>(),"size of the median filter")
        ("STWarp.useAdvancedMedian",po::value<bool>(),"edge preserving median filter")
        ("STWarp.useFlow",po::value<bool>(),"use optical flow cues")
        ("STWarp.limitUpdate",po::value<bool>(),"limit warp update to range 1")
        ("STWarp.gamma",po::value<double>(),"mixing ratio of normal vs data term")

        ("NNF.patchSize_space",po::value<int>(),"spatial patch extent")
        ("NNF.patchSize_time",po::value<int>(),"temporal patch extent")
        ("NNF.propagationIterations",po::value<int>(),"number of correspondence propagation iterations")
        
        ("Render.splatSize",po::value<int>(),"splat size in pixels")
        ("Render.renderSmoothing",po::value<bool>(),"filter the warped field")
    ;

    // Read config file
    po::variables_map vm;
    bool allow_unregistered=true;
    fs::ifstream f(path);
    po::store(
            po::parse_config_file(f, desc, allow_unregistered),
            vm);
    notify(vm);
    
    // Fill parameters
    if(vm.count("STWarp.autoLevels")) 
        autoLevels = vm["STWarp.autoLevels"].as<bool>();
    if(vm.count("STWarp.minPyrSize")) 
        minPyrSize = vm["STWarp.minPyrSize"].as<int>();
    if(vm.count("STWarp.pyrSpacing")) 
        pyrSpacing = vm["STWarp.pyrSpacing"].as<double>();
    if(vm.count("STWarp.warpIterations")) 
        warpIterations = vm["STWarp.warpIterations"].as<int>();
    if(vm.count("STWarp.solverIterations")) 
        solverIterations = vm["STWarp.solverIterations"].as<int>();
    if(vm.count("STWarp.output")) 
        outputPath = fs::path(vm["STWarp.output"].as<std::string>());
    if(vm.count("STWarp.lambdaSUV")) 
        lambda[0] = vm["STWarp.lambdaSUV"].as<double>();
    if(vm.count("STWarp.lambdaTUV")) 
        lambda[1] = vm["STWarp.lambdaTUV"].as<double>();
    if(vm.count("STWarp.lambdaSW")) 
        lambda[2] = vm["STWarp.lambdaSW"].as<double>();
    if(vm.count("STWarp.lambdaTW")) 
        lambda[3] = vm["STWarp.lambdaTW"].as<double>();
    if(vm.count("STWarp.lambdaMult")) {
        double m = vm["STWarp.lambdaMult"].as<double>();
        for (int i = 0; i < 4; ++i) {
            lambda[i] *= m;
        }
    }
    if(vm.count("STWarp.localSmoothness")) 
        localSmoothness = vm["STWarp.localSmoothness"].as<double>();
    if(vm.count("STWarp.localSmoothnessPrior")) 
        localSmoothnessPrior = vm["STWarp.localSmoothnessPrior"].as<double>();
    if(vm.count("STWarp.divPrior")) 
        divPrior = vm["STWarp.divPrior"].as<double>();
    if(vm.count("STWarp.occlusionPrior")) 
        occlusionPrior = vm["STWarp.occlusionPrior"].as<double>();
    if(vm.count("STWarp.useOcclusion")) 
        useOcclusion = vm["STWarp.useOcclusion"].as<bool>();
    if(vm.count("STWarp.useColor")) 
        useColor = vm["STWarp.useColor"].as<bool>();
    if(vm.count("STWarp.useFeatures")) 
        useFeatures = vm["STWarp.useFeatures"].as<bool>();
    if(vm.count("STWarp.bypassTimeWarp")) 
        bypassTimeWarp = vm["STWarp.bypassTimeWarp"].as<bool>();
    if(vm.count("STWarp.decoupleRegularization")) 
        decoupleRegularization = vm["STWarp.decoupleRegularization"].as<int>();
    if(vm.count("STWarp.debug")) 
        debug = vm["STWarp.debug"].as<bool>();
    if(vm.count("STWarp.c")) 
        c = vm["STWarp.c"].as<double>();
    if(vm.count("STWarp.useAdvancedMedian")) 
        useAdvancedMedian = vm["STWarp.useAdvancedMedian"].as<bool>();
    if(vm.count("STWarp.useFlow")) 
        useFlow = vm["STWarp.useFlow"].as<bool>();
    if(vm.count("STWarp.medfiltSize")) 
        medfiltSize = vm["STWarp.medfiltSize"].as<int>();
    if(vm.count("STWarp.limitUpdate")) 
        limitUpdate = vm["STWarp.limitUpdate"].as<bool>();
    if(vm.count("STWarp.gamma")) 
        gamma = vm["STWarp.gamma"].as<double>();

    if(vm.count("NNF.patchSize_space")) 
        patchSize_space = vm["NNF.patchSize_space"].as<int>();
    if(vm.count("NNF.patchSize_space")) 
        patchSize_time = vm["NNF.patchSize_time"].as<int>();
    if(vm.count("NNF.patchSize_space")) 
        propagationIterations = vm["NNF.propagationIterations"].as<int>();

    if(vm.count("Render.splatSize")) 
        splatSize = vm["Render.splatSize"].as<int>();
    if(vm.count("Render.renderSmoothing")) 
        renderSmoothing = vm["Render.renderSmoothing"].as<bool>();
}
