#ifndef STWARPPARAMS_HPP_YCEJNHV1
#define STWARPPARAMS_HPP_YCEJNHV1

#include <string>


/**
 * Parameters structure for the space-time correspondence algorithm.
 */
class STWarpParams 
{
public:
    std::string     name;
    std::string     outputPath;
    bool            useColor;
    bool            useFeatures;
    bool            bypassTimeWarp;
    bool            autoLevels;
    int             minPyrSize;
    double          pyrSpacing;
    int             pyrLevels;
    int             warpIterations;
    int             solverIterations;
    double          eps;
    double          lambda[4];
    int             decoupleRegularization;
    double          localSmoothness;
    double          localSmoothnessPrior;
    double          divPrior;
    double          occlusionPrior;
    bool            useOcclusion;
    bool            limitUpdate;
    bool            debug;
    double          c;
    bool            useAdvancedMedian;
    bool            useFlow;
    int            medfiltSize;
    double         gamma;

    int patchSize_space;
    int patchSize_time;
    int propagationIterations;

    int splatSize;
    bool renderSmoothing;

    STWarpParams() :
        name("default"),
        outputPath("../output"),
        useColor(false),
        useFeatures(false),
        bypassTimeWarp(false),
        autoLevels(true),
        minPyrSize(16),
        pyrSpacing(1.25),
        pyrLevels(1),
        warpIterations(3),
        solverIterations(30),
        eps(1e-3),
        decoupleRegularization(0),
        localSmoothness(1.5),
        localSmoothnessPrior(2),
        divPrior(0.3),
        occlusionPrior(20),
        useOcclusion(0),
        limitUpdate(false),
        debug(false),
        c(1),
        useAdvancedMedian(true),
        useFlow(true),
        medfiltSize(5),
        gamma(1),

        patchSize_space(7),
        patchSize_time(5),
        propagationIterations(5),

        splatSize(3),
        renderSmoothing(true)
    {
        lambda[0] = 1; // uv spatial reg
        lambda[1] = 1; // uv time reg
        lambda[2] = 1; // w spatial reg
        lambda[3] = 1; // w time reg
        for (int i = 0; i < 4; ++i) {
            lambda[i] *= .3;
        }
    }

private:
};

#endif /* end of include guard: STWARPPARAMS_HPP_YCEJNHV1 */