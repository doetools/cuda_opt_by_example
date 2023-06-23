#include "data_structure.h"

template <class T, size_t WARP_SIZE = 32>
__global__ void matrix_multiplication_noncoalesce(T const *a, T const *b, T *c,
                                                  const int M, const int N) {
  size_t i{blockIdx.x * blockDim.x + threadIdx.x};
  size_t j{blockIdx.y * blockDim.y + threadIdx.y};

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

  T total{0};

  if (i >= M || j >= M)
    return;

  for (int k = 0; k < N; k++) {
    total += a[N * i + k] * b[k * M + j];
  }

  // dot product of i row of matrix a and j colum of b
  c[i * N + j] = total;
}
