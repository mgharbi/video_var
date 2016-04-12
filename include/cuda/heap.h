#ifndef HEAP_H_2XIEPCJH
#define HEAP_H_2XIEPCJHHEAP_H_2XIEPCJH


#include "cuda/utils.h"

namespace cuda
{

namespace heap
{

//! moves an element down the heap until all children are smaller than the element
//! if c is a less-than comparator, it do this until all children are larger
template <class GreaterThan, class RandomAccessIterator>
__host__ __device__ void
sift_down( RandomAccessIterator array, size_t begin, size_t length, GreaterThan c = GreaterThan() )
{

    while( 2*begin+1 < length ) {
        size_t left = 2*begin+1;
        size_t right = 2*begin+2;
        size_t largest=begin;
        if((left < length)&& c(array[left], array[largest]) ) largest=left;

        if((right < length)&& c(array[right], array[largest]) ) largest=right;

        if( largest != begin ) {
            cuda::swap( array[begin], array[largest] );
            begin=largest;
        }
        else return;
    }
}

//! creates a max-heap in the array beginning at begin of length "length"
//! if c is a less-than comparator, it will create a min-heap
template <class GreaterThan, class RandomAccessIterator>
__host__ __device__ void
make_heap( RandomAccessIterator begin, size_t length, GreaterThan c = GreaterThan() )
{
    int i=length/2-1;
    while( i>=0 ) {
        sift_down( begin, i, length, c );
        i--;
    }
}


//! verifies if the array is a max-heap
//! if c is a less-than comparator, it will verify if it is a min-heap
template <class GreaterThan, class RandomAccessIterator>
__host__ __device__ bool
is_heap( RandomAccessIterator begin, size_t length, GreaterThan c = GreaterThan() )
{
    for( unsigned i=0; i<length; i++ ) {
        if((2*i+1 < length)&& c(begin[2*i+1],begin[i]) ) return false;
        if((2*i+2 < length)&& c(begin[2*i+2],begin[i]) ) return false;
    }
    return true;
}

// template <class GreaterThan, class RandomAccessIterator>
// __host__ __device__ bool
// heap_insert( RandomAccessIterator begin, size_t length, GreaterThan c = GreaterThan() )
// {
//     // begin[length-1]
//     unsigned i = length-1;
//     while(i > 1 && c(begin[i/2], begin[i])) {
//         swap(begin[i], begin[i/2]);
//         i = i/2;
//     }
//     // for( unsigned i=length-1; i<length; i++ ) {
//     //     if((2*i+1 < length)&& c(begin[2*i+1],begin[i]) ) return false;
//     //     if((2*i+2 < length)&& c(begin[2*i+2],begin[i]) ) return false;
//     // }
//     return true;
// }


//! moves an element down the heap until all children are smaller than the element
//! if c is a less-than comparator, it do this until all children are larger
template <class GreaterThan, class RandomAccessIterator, class RandomAccessIterator2>
__host__ __device__ void
sift_down( RandomAccessIterator key, RandomAccessIterator2 value, size_t begin, size_t length, GreaterThan c = GreaterThan() )
{

    while( 2*begin+1 < length ) {
        size_t left = 2*begin+1;
        size_t right = 2*begin+2;
        size_t largest=begin;
        if((left < length)&& c(key[left], key[largest]) ) largest=left;

        if((right < length)&& c(key[right], key[largest]) ) largest=right;

        if( largest != begin ) {
            cuda::swap( key[begin], key[largest] );
            cuda::swap( value[begin], value[largest] );
            begin=largest;
        }
        else return;
    }
}

//! creates a max-heap in the array beginning at begin of length "length"
//! if c is a less-than comparator, it will create a min-heap
template <class GreaterThan, class RandomAccessIterator, class RandomAccessIterator2>
__host__ __device__ void
make_heap( RandomAccessIterator key,  RandomAccessIterator2 value, size_t length, GreaterThan c = GreaterThan() )
{
    int i=length/2-1;
    while( i>=0 ) {
        sift_down( key, value, i, length, c );
        i--;
    }
}

}

} // namespace cuda

#endif /* end of include guard: HEAP_H_2XIEPCJHHEAP_H_2XIEPCJH */
