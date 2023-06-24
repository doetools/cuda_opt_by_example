# 001. Global Memory Coalescing

Date: 2023-06-23

## Context

There are many factors associated with CUDA memory that can be optimized, such as bandwith and reading/writing efficiency. There are two key concepts that need to be introduced before discussing the memory coalescing. The first concept is the CUDA architecture. From the standpoint of memory, there are two big memories that are on the GPU but not on the streaming multiprocessors (SM), called global memory and constant memory. The global memory, which is in DRAM, by default is cached in L1 and could be cached in L2 by passing `-dlcm=cg` to nvcc compiler. Compared to L1 cache and shared memory on SM, global memory, especially when moving a large chunk of data (load and store) is slow. Shared memory, which is a on-chip memory shared by a block of threads, is significantly faster but relatively small in capacity.

When in execution, the blocks are further divided into warps; a warp, which is a primitive scheduling unit, consists of 32 threads and performs SIMD (single instruction and multiple data). Each warp each time requests 128 byte data, or, 32 4-byte word. If the requested data has consecutive address on global memory or L1/L2 cache, then coalescing memory read can be performed to minimize the transaction and reduce bandwidth consumption (or improve bus utilization). Conversely, if the requested data is scattered and misaligned across mulitple cache-lines, it may take more cycles for a warp to load and store the data. This will lead to bandwidth waste.

## Experiments

Below are three examples to show the difference between coalesced and non-coalesced memory reading.

In below example, each warp (can be visualed as a row of threads in a block) reading part of a row in matrix `a` and increments the value by 1. As the each row has consecutive addresses, the memory reading is coalesced.

```cpp cuda
template<typename T>
__global__ void add(__device__ T* a, const int N){
    size_t j = blockIdx.x * blockDim.x + threadIdx.x;
    size_t i = blockIdx.y * blockDim.y + threadIdx.y;

    a[i*N + j] += 1;
}
```

In below example, we switched the `i` and `j`. In this case, the map of threads has to be transposed to be agreeing with matrix `a`. For example, the thread `(1, 2)` (first row and second column) is corresponding to `a[2,1]`. Although the results will be the same as the above kernel, the speed is supposedly slower, as reading data from the global memory is less effcient. Why? The reason is because now threads in a warp is now reading part of a column of matrix `a`, in which two adjacent data request does not have consecutive addresses (suppose the matrix is row based).

```cpp cuda
template<typename T>
__global__ void add(__device__ T* a, const int N){
    size_t i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    size_t j = blockIdx.y * blockDim.y + threadIdx.y + 1;

    a[i*N + j] += 1;
}
```

For a more thorough experiment, check out [memory_coalesce.cu]("./memory_coalesce.cu").

## Findings

Based on a experiment of multiplication of two matrices of 1024 by 1024, the kernel with coalesced memory reading is about 2 times faster.

## References

1. https://developer.nvidia.com/blog/how-access-global-memory-efficiently-cuda-c-kernels/
