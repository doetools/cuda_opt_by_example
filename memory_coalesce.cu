#include "data_structure.h"

// T const* a should not be confused with T* const a
// the first one says T is a constant while the second one says
// the pointer is a constant
// ALWAYS READ BACKWARD (<---), i.e., float const *, reads
// a pointer to a constant float
template <class T, size_t block_size=32>
__global__ void dumb_kernel(float const *a, const int N)
{
    size_t gid{blockIdx.x * blockDim.x + threadIdx.x};
    if (gid < N)
        printf("the value is %f\n", a[gid]);
}


int main()
{
    DataBuffer<FLOAT> a = DataBuffer<FLOAT>(5, 5);

    dim3 const blocks{32};
    dim3 const grids{1};

    dumb_kernel<FLOAT><<<grids, blocks>>>(a.d_data, a.size);

    // std::cout << a.c_data.size() << std::endl;
    // for (auto i : a.c_data)
    //     std::cout << a.size << std::endl;

    return 0;
}