#include "data_structure.h"
// T const* a should not be confused with T* const a
// the first one says T is a constant while the second one says
// the pointer is a constant
// ALWAYS READ BACKWARD (<---), i.e., float const *, reads
// a pointer to a constant float
template <class T, size_t block_size = 32>
__global__ void matrix_multiplication_noncoalesce(T const *a, T const *b, T *c,
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

template <typename T>
vector<T> matrix_multiplication(vector<T> const &a, vector<T> const &b,
                                const int M, const int N) {

  vector<T> c(M * M, T(0));

  T total{T(0)};

  for (int i = 0; i < M; i++) {
    for (int j = 0; j < M; j++) {
      total = T(0);
      for (int k = 0; k < N; k++) {
        total += a[i * N + k] * b[k * M + j];
      }
      c[i * M + j] = total;
    }
  }

  return c;
}

template <class T> T max_difference(vector<T> &target, vector<T> &src) {

  vector<float> diff(target.size(), T(0.0));

  for (int i = 0; i < target.size(); i++) {
    diff[i] = fabs(target[i] - src[i]);
  }

  // max_element returns a pointer.
  T max_diff = *max_element(diff.begin(), diff.end());

  return max_diff;
}