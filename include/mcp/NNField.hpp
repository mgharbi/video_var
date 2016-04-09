#ifndef NNFIELD_HPP_RCFNREA6
#define NNFIELD_HPP_RCFNREA6

#include <limits>
#include <tuple>
#include <unordered_set>

#include "mcp/WarpingField.hpp"
#include "video/Video.hpp"

typedef float nnf_data_t;

typedef struct NNFieldParams {
    NNFieldParams(int it = 1, int psz_space = 5, int psz_time = 5,
            int knn = 1, int threads = 16, int verbosity = 1, int jump_flood_step = 8) 
        : propagation_iterations(it), 
          patch_size_space(psz_space),
          patch_size_time(psz_time),
          knn(knn),
          threads(threads),
          verbosity(verbosity),
          jump_flood_step(jump_flood_step)
    {}
    int propagation_iterations;
    int patch_size_space;
    int patch_size_time;
    int knn;
    int threads;
    int verbosity;
    int jump_flood_step;
} NNFieldParams;

typedef struct NNFieldOutput {
    NNFieldOutput(int h, int w, int nF, int knn) 
        : nnf(h, w,  nF, 3*knn),
        error(h, w,  nF, knn)
    {
    }
    Video<int> nnf;
    Video<nnf_data_t> error;
} NNFieldReturnOutput ;

typedef std::tuple<nnf_data_t,int,int,int> Match; // cost,x,y,t

typedef struct MatchHash
{
    std::size_t operator()(const Match& p) const
    {
        return std::get<1>(p) ^ std::get<2>(p) ^ std::get<3>(p);
    }
} MatchHash;

typedef struct MatchEqualTo {
    bool operator() (const Match& a, const Match& b)  const {
        bool ret = std::get<1>(a) == std::get<1>(b);
        ret &= (std::get<2>(a) == std::get<2>(b)); 
        ret &= (std::get<3>(a) == std::get<3>(b)); 
        return ret;
    }
} MatchEqualTo;


// typedef std::unordered_set<Match, MatchHash> MatchSet;
typedef std::unordered_set<Match, MatchHash, MatchEqualTo> MatchSet;


class NNField
{
public:
    NNField (const Video<nnf_data_t> *video, const Video<nnf_data_t> *database, 
        NNFieldParams params = NNFieldParams() ) 
        : video_(video), database_(database), params_(params) {};
    virtual ~NNField ();

    NNFieldOutput compute(); // k-nearest neighbors
    NNFieldOutput compute_gpu(); // k-nearest neighbors

private:
    const Video<nnf_data_t> *video_;
    const Video<nnf_data_t> *database_;

    NNFieldParams params_;
    int nn_offset_;

    nnf_data_t getPatchCost(const Video<nnf_data_t> &A,const Video<nnf_data_t> &B, int y_a, int x_a, int t_a, int y_b, int x_b, int t_b);
    void improve_knn(const Video<nnf_data_t> &A,const Video<nnf_data_t> &B,int y_a, int x_a, int t_a,
        vector<Match> &current_best,
        MatchSet &all_matches,
        int y_p, int x_p, int t_p);
};

#endif /* end of include guard: NNFIELD_HPP_RCFNREA6 */

