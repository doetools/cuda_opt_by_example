#include "../../cuda/data_structure.h"

template <typename T>
__host__ __device__ T convolute(T *a, T *b, const int N)
{
    T total{0};

    for (int i = 0; i < N; i++)
    {
        total += a[i] * b[N - 1 - i];
    }

    return T(total);
}

// kernels
__device__ const FLOAT KERNEL_X[9] = {1.0, 0.0, -1.0, 2.0, 0.0, -2.0, 1.0, 0.0, -1.0};
__device__ const FLOAT KERNEL_Y[9] = {-1.0, -2.0, -1.0, 0, 0.0, 0.0, 1.0, 2.0, 1.0};

template <class T, const int WARP_SIZE = 32>
__global__ void sobel_convolute_naive(const T *img, T *new_img, const size_t M, const size_t N)
{

    size_t gj{blockIdx.x * blockDim.x + threadIdx.x};
    size_t gi{blockIdx.y * blockDim.y + threadIdx.y};

    size_t j = threadIdx.x;
    size_t i = threadIdx.y;

    int KERNEL_SIZE{3};
    int KERNEL_LEN{9};

    // a 32 by 32 tile
    __shared__ T img_tile[WARP_SIZE + 2][WARP_SIZE + 2];

    // read data to shared memory
    // there are 2 rows/columns of overlapping between two tiles
    if (blockIdx.x == 0)
        img_tile[i][j] = img[gi * N + gj];
    else
        img_tile[i][j] = img[gi * N + gj - 2];

    if (blockIdx.y == 0)
        img_tile[i][j] = img[gi * N + gj];
    else
        img_tile[i][j] = img[(gi - 1) * N + gj];

    __syncthreads();

    // skip the threads on the boundary
    if (i == 0 || i == WARP_SIZE - 1)
        return;
    if (j == 0 || j == WARP_SIZE - 1)
        return;

    // calculate sobel
    int index;
    int increment_i = 0, increment_j = 0;

    FLOAT total_x = 0;
    for (index = 0; index < KERNEL_LEN; index++)
    {
        increment_i = index / KERNEL_SIZE;
        increment_j = index % KERNEL_SIZE;
        total_x += img_tile[i - 1 + increment_i][j - 1 + increment_j] * KERNEL_X[KERNEL_LEN - 1 - index];
    }

    FLOAT total_y = 0;
    for (index = 0; index < KERNEL_LEN; index++)
    {
        increment_i = index / KERNEL_SIZE;
        increment_j = index % KERNEL_SIZE;
        total_y += img_tile[i - 1 + increment_i][j - 1 + increment_j] * KERNEL_Y[KERNEL_LEN - 1 - index];
    }

    // save result to new_img of M-2 by N-2
    new_img[gi * N + gj] = sqrtf(total_x * total_x + total_y * total_y);
}