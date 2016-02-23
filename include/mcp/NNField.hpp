#ifndef NNFIELD_HPP_RCFNREA6
#define NNFIELD_HPP_RCFNREA6

#include <limits>
#include <tuple>
#include <unordered_map>

#include "mcp/WarpingField.hpp"
#include "video/Video.hpp"

typedef struct NNFieldParams {
    NNFieldParams(int it = 1, int psz_space = 5, int psz_time = 5, int knn = 1) 
        : propagation_iterations(it), 
          patch_size_space(psz_space),
          patch_size_time(psz_time),
          knn(knn) {}
    int propagation_iterations;
    int patch_size_space;
    int patch_size_time;
    int knn;
} NNFieldParams;

typedef std::tuple<float,int,int,int> Match;
typedef std::tuple<int,int,int> PatchCoord;

struct PatchCoordHash : public std::unary_function<PatchCoord, std::size_t>
{
    std::size_t operator()(const PatchCoord& p) const
    {
        return std::get<0>(p) ^ std::get<1>(p) ^ std::get<2>(p);
    }
};

typedef std::unordered_map<PatchCoord,bool, PatchCoordHash> PatchCoordHashMap;


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

    float getPatchCost(const IVideo &A,const IVideo &B, int y_a, int x_a, int t_a, int y_b, int x_b, int t_b);
    void improve_guess(const IVideo &A,const IVideo &B,int y_a, int x_a, int t_a,
        int &y_best, int &x_best, int &t_best, float &cost,
        int y_p, int x_p, int t_p);
    void improve_knn(const IVideo &A,const IVideo &B,int y_a, int x_a, int t_a,
        vector<Match> &current_best,
        PatchCoordHashMap &all_matches,
        int y_p, int x_p, int t_p);
};

#endif /* end of include guard: NNFIELD_HPP_RCFNREA6 */

