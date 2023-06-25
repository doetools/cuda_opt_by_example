#include "../../cuda_opt_basics/data_structure.h"

__device__ const int KERNEL_X[9] = {-1, 0, 1, -2, 0, 2, -1, 0, 1};
__device__ const int KERNEL_Y[9] = {1, 2, 1, 0, 0, 0, -1, -2, -1};

template <class T>
__global__ void convolute(T *image)
{

    size_t gj{blockIdx.x * blockDim.x + threadIdx.x};

    size_t gi{blockIdx.y * blockDim.y + threadIdx.y};

    size_t j = threadIdx.x;
    size_t i = threadIdx.y;
}