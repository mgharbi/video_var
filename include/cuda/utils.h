#ifndef CUDA_UTILS_H_UJ6F9SZA
#define CUDA_UTILS_H_UJ6F9SZA


#define CUDA_KERNEL_LOOP(i, n) \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
       i < (n); \
       i += blockDim.x * gridDim.x)

const int MCP_GPU_NUM_THREADS = 512;
inline int MCP_GPU_GET_BLOCKS(const int N) {
  return (N + MCP_GPU_NUM_THREADS - 1) / MCP_GPU_NUM_THREADS;
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
