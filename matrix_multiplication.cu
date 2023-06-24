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
