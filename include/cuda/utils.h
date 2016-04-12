#ifndef CUDA_UTILS_H_UJ6F9SZA
#define CUDA_UTILS_H_UJ6F9SZA


#define CUDA_KERNEL_LOOP(i, n) \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
       i < (n); \
       i += blockDim.x * gridDim.x)


// NOTE: tw*th*td < 512 (or 1024 on new gpu)
const int TILE_W = 16;
const int TILE_H = 16;
const int TILE_D = 2;
const int GPU_THREADS = 1024;
inline int GPU_GET_BLOCKS(const int N) {
  return (N + GPU_THREADS - 1) / GPU_THREADS;
}

namespace cuda
{
template <class T>
__device__ __host__ void swap( T& x, T& y )
{
    T t=x;
    x=y;
    y=t;
}


} // namespace cuda


#endif /* end of include guard: CUDA_UTILS_H_UJ6F9SZA */
