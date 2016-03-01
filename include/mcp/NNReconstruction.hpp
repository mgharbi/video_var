#ifndef NNRECONSTRUCTION_HPP_EAUCSDZ4
#define NNRECONSTRUCTION_HPP_EAUCSDZ4

#include "video/Video.hpp"

typedef struct NNReconstructionParams {
    NNReconstructionParams(int psz_space = 5, int psz_time = 5, int knn = 1, int threads = 16) 
        : 
          patch_size_space(psz_space),
          patch_size_time(psz_time),
          knn(knn),
          threads(threads)
    {}
    int patch_size_space;
    int patch_size_time;
    int knn;
    int threads;
} NNReconstructionParams;

class NNReconstruction
{
public:
    NNReconstruction(const IVideo *db, const Video<int> *nnf, 
            const Video<float> *w, NNReconstructionParams p)
        : db_(db), nnf_(nnf), w_(w), params_(p) {};

    virtual ~NNReconstruction() {};

    IVideo reconstruct();


private:
    const IVideo *db_;
    const Video<int> *nnf_;
    const Video<float> *w_;
    NNReconstructionParams params_;
};


#endif /* end of include guard: NNRECONSTRUCTION_HPP_EAUCSDZ4 */

