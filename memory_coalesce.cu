#include "data_structure.h"
#include "matrix_multiplication.cu"

template <class T> constexpr T ceiling_div(T a, T b) { return (a + b - 1) / b; }

int main() {
  const size_t M{1 << 5};
  const size_t N{1 << 5};
  const size_t WARP_SIZE{32};

  DataBuffer<FLOAT> a = DataBuffer<FLOAT>(M, N);
  DataBuffer<FLOAT> b = DataBuffer<FLOAT>(N, M);
  DataBuffer<FLOAT> c = DataBuffer<FLOAT>(M, M, false);

  dim3 const blocks{WARP_SIZE, WARP_SIZE};
  dim3 const grids{ceiling_div<size_t>(M, WARP_SIZE),
                   ceiling_div<size_t>(M, WARP_SIZE)};

  matrix_multiplication_noncoalesce<FLOAT>
      <<<grids, blocks>>>(a.d_data, b.d_data, c.d_data, M, N);

  c.copy_to_host();

  vector<FLOAT> c_ref = matrix_multiplication<FLOAT>(a.c_data, b.c_data, M, M);

  cout << max_difference(c_ref, c.c_data) << endl;

  return 0;
}
