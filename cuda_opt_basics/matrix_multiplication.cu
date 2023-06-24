#include "data_structure.h"

template <class T, size_t WARP_SIZE = 32>
__global__ void matrix_multiplication_noncoalesce(T const *a, T const *b, T *c,
                                                  const int M, const int N) {
  size_t i{blockIdx.x * blockDim.x + threadIdx.x};
  size_t j{blockIdx.y * blockDim.y + threadIdx.y};

  // notice that the threads have to be transposed before mapping to c.

  // All threads of a warp access a part of column of a
  // since the global memory of columns are not consecutive
  // so the global memory access of a warp is not coalescing

  // All threads of a warp access the same value of b, a broadcast
  // within the warp will take place.

  T total{0};

  if (i >= M || j >= M)
    return;

  for (int k = 0; k < N; k++) {
    total += a[N * i + k] * b[k * M + j];
  }

  // dot product of i row of matrix a and j colum of b
  c[i * N + j] = total;
}

template <class T, size_t WARP_SIZE = 32>
__global__ void matrix_multiplication_coalesce(T const *a, T const *b, T *c,
                                               const int M, const int N) {
  size_t j{blockIdx.x * blockDim.x + threadIdx.x};
  size_t i{blockIdx.y * blockDim.y + threadIdx.y};

  // notice that compared to the above kernel, the assignment to i and j
  // are swapped. The threads are able to directly mapped to c.

  // now all threads of a warp access the same value of a.

  // all threads if a waro access part of a row in b, which has consecutive
  // addresses, and therefore a coalesced memory access will take place.

  T total{0};

  if (i >= M || j >= M)
    return;

  for (int k = 0; k < N; k++) {
    total += a[N * i + k] * b[k * M + j];
  }

  // dot product of i row of matrix a and j colum of b
  c[i * N + j] = total;
}

template <class T, size_t WARP_SIZE = 32>
__global__ void matrix_multiplication_shared_memory(T const *a, T const *b,
                                                    T *c, const int M,
                                                    const int N) {
  // global thread id
  size_t gj{blockIdx.x * blockDim.x + threadIdx.x};
  size_t gi{blockIdx.y * blockDim.y + threadIdx.y};

  // thread id within block
  size_t j{threadIdx.x};
  size_t i{threadIdx.y};

  // shared memory or tile
  __shared__ T a_shared[WARP_SIZE][WARP_SIZE];
  __shared__ T b_shared[WARP_SIZE][WARP_SIZE];

  T total{0};

  if (gi >= M || gj >= M)
    return;

  for (int k = 0; k < (N + WARP_SIZE - 1) / WARP_SIZE; k++) {
    // read data to tile
    // a[gi][j + k * WARP_SIZE], b[i + k * WARP_SIZE]
    // a is M by N and b is N by M
    a_shared[i][j] = a[gi * N + j + k * WARP_SIZE];
    b_shared[i][j] = b[(i + k * WARP_SIZE) * M + gj];

    // wait until all threads finish
    __syncthreads();

    // matrix mulplication over the tile
    for (int id = 0; id < WARP_SIZE; id++) {
      total += a_shared[i][id] * b_shared[id][j];
    }

    // wait until all threads finish
    __syncthreads();
  }

  // dot product of i row of matrix a and j colum of b
  c[gi * N + gj] = total;
}
