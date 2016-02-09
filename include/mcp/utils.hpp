/* --------------------------------------------------------------------------
 * File:    utils.hpp
 * Author:  Michael Gharbi <gharbi@mit.edu>
 * Created: 2014-04-08
 * --------------------------------------------------------------------------
 * 
 * 
 * 
 * ------------------------------------------------------------------------*/

#include <cmath>
#include <vector>
#include <iostream>

using namespace std;

template <class T, class T2>
T weightedMedian(const vector<T> &values, const vector<T2> &weight) {
    // NOTE: assumes sum of weights is 1
    vector<size_t> idx(values.size());
    for (size_t i = 0; i != idx.size(); ++i) idx[i] = i;
    sort(idx.begin(), idx.end(),
            [&values](size_t i1, size_t i2) {return values[i1] < values[i2];});

    int k = 0;
    double sum = 1 - weight[idx[0]]; // sum is the total weight of all `x[i] > x[k]`

    while(sum > .5)
    {
        ++k;
        sum -= weight[idx[k]];
    }

    return values[idx[k]];
}
