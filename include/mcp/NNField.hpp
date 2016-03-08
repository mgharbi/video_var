#ifndef NNFIELD_HPP_RCFNREA6
#define NNFIELD_HPP_RCFNREA6

#include <limits>
#include <tuple>
#include <unordered_set>

#include "mcp/WarpingField.hpp"
#include "video/Video.hpp"

typedef struct NNFieldParams {
    NNFieldParams(int it = 1, int psz_space = 5, int psz_time = 5,
            int knn = 1, int threads = 16, int verbosity = 1) 
        : propagation_iterations(it), 
          patch_size_space(psz_space),
          patch_size_time(psz_time),
          knn(knn),
          threads(threads),
          verbosity(verbosity)
    {}
    int propagation_iterations;
    int patch_size_space;
    int patch_size_time;
    int knn;
    int threads;
    int verbosity;
} NNFieldParams;

typedef std::tuple<int,int,int,int> Match; // cost,x,y,t

struct MatchHash : public std::unary_function<Match, std::size_t>
{
    std::size_t operator()(const Match& p) const
    {
        return std::get<1>(p) ^ std::get<2>(p) ^ std::get<3>(p);
    }
};

struct MatchEqualTo : std::binary_function<Match,Match,bool> {
    bool operator() (const Match& a, const Match& b)  {
        bool ret = std::get<1>(a) == std::get<1>(b);
        ret &= (std::get<2>(a) == std::get<2>(b)); 
        ret &= (std::get<3>(a) == std::get<3>(b)); 
        return ret;
    }
};


typedef std::unordered_set<Match, MatchHash, MatchEqualTo> MatchSet;


class NNField
{
public:
    NNField (const IVideo *video, const IVideo *database, 
        NNFieldParams params = NNFieldParams() ) 
        : video_(video), database_(database), params_(params) {};
    virtual ~NNField ();

    Video<int> compute(); // k-nearest neighbors

private:
    const IVideo *video_;
    const IVideo *database_;

    NNFieldParams params_;
    int nn_offset_;

    int getPatchCost(const IVideo &A,const IVideo &B, int y_a, int x_a, int t_a, int y_b, int x_b, int t_b);
    void improve_guess(const IVideo &A,const IVideo &B,int y_a, int x_a, int t_a,
        int &y_best, int &x_best, int &t_best, int &cost,
        int y_p, int x_p, int t_p);
    void improve_knn(const IVideo &A,const IVideo &B,int y_a, int x_a, int t_a,
        vector<Match> &current_best,
        MatchSet &all_matches,
        int y_p, int x_p, int t_p);
};

#endif /* end of include guard: NNFIELD_HPP_RCFNREA6 */

