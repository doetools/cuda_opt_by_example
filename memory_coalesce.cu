#include "data_structure.h"
#include "matrix_multiplication.cu"
#include "utility.cpp"
#include "utility.cu"

template <class T> constexpr T ceiling_div(T a, T b) { return (a + b - 1) / b; }

int main() {
  const size_t M{1 << 10};
  const size_t N{1 << 10};
  const size_t WARP_SIZE{32};

  cudaStream_t stream;

  DataBuffer<FLOAT> a = DataBuffer<FLOAT>(M, N);
  DataBuffer<FLOAT> b = DataBuffer<FLOAT>(N, M);
  DataBuffer<FLOAT> c1 = DataBuffer<FLOAT>(M, M, false);
  DataBuffer<FLOAT> c2 = DataBuffer<FLOAT>(M, M, false);
  DataBuffer<FLOAT> c3 = DataBuffer<FLOAT>(M, M, false);

  dim3 const blocks{WARP_SIZE, WARP_SIZE};
  dim3 const grids{ceiling_div<size_t>(M, WARP_SIZE),
                   ceiling_div<size_t>(M, WARP_SIZE)};

  // create two wrapped binding functions
  // use the lambda format, where & is for using references of all
  // internal vairables, v.s., = for using copies
  std::function<void(cudaStream_t)> const launch_noncoalesce{
      [&](cudaStream_t stream) {
        matrix_multiplication_noncoalesce<FLOAT>
            <<<grids, blocks, 0, stream>>>(a.d_data, b.d_data, c1.d_data, M, N);
      }};

  std::function<void(cudaStream_t)> const launch_coalesce{
      [&](cudaStream_t stream) {
        matrix_multiplication_coalesce<FLOAT>
            <<<grids, blocks, 0, stream>>>(a.d_data, b.d_data, c2.d_data, M, N);
      }};

  std::function<void(cudaStream_t)> const launch_shared_memory{
      [&](cudaStream_t stream) {
        matrix_multiplication_shared_memory<FLOAT>
            <<<grids, blocks, 0, stream>>>(a.d_data, b.d_data, c3.d_data, M, N);
      }};

  // creaete stream
  cudaStreamCreate(&stream);
  FLOAT time_noncoalesce = measure_performance(launch_noncoalesce, stream, 10);
  FLOAT time_coalesce = measure_performance(launch_coalesce, stream, 10);
  FLOAT time_shared_memory =
      measure_performance(launch_shared_memory, stream, 10);

  cout << "time cost of noncoalescing memory read is " << time_noncoalesce
       << " ms" << endl;

  cout << "time cost of coalescing memory read is " << time_coalesce << " ms"
       << endl;

  cout << "time cost of shared memory read is " << time_shared_memory << " ms"
       << endl;

  // destroy stream
  cudaStreamDestroy(stream);

  c1.copy_to_host();
  c2.copy_to_host();
  c3.copy_to_host();

  vector<FLOAT> c_ref = matrix_multiplication<FLOAT>(a.c_data, b.c_data, M, M);

  cout << "difference between cpu and gpu is "
       << max_difference(c_ref, c3.c_data) << endl;

  return 0;
}
