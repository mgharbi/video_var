/* --------------------------------------------------------------------------
 * File:    VideoExceptions.hpp
 * Author:  Michael Gharbi <gharbi@mit.edu>
 * Created: 2014-01-29
 * --------------------------------------------------------------------------
 * 
 * Exceptions arising while using the video library.
 * 
 * ------------------------------------------------------------------------*/


#include <exception>

using namespace std;

class IncorrectSizeException : public exception
{
    virtual const char* what() const throw(){
        return "Operation cannot be performed: sizes do not match.";
    }
};
