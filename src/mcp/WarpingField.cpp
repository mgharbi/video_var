/* --------------------------------------------------------------------------
 * File:    WarpingField.cpp
 * Author:  Michael Gharbi <gharbi@mit.edu>
 * Created: 2014-01-29
 * --------------------------------------------------------------------------
 * 
 * Sub-class of Video implementing the specificities of the ST warping field
 * 
 * ------------------------------------------------------------------------*/

#include <cstdlib>
#include <cmath>
#include <fstream>

#include "mcp/WarpingField.hpp"



using namespace std;
typedef unsigned char uchar;


template <class T>
WarpingField<T>::~WarpingField() {
}



template class WarpingField<float>;
template class WarpingField<double>;
