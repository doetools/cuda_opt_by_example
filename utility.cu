#include "data_structure.h"

// template <class T>
FLOAT measure_performance(std::function<void(cudaStream_t)> bound_function,
                          cudaStream_t stream, int num_repeats = 100,
                          int num_warmups = 100)
{
    cudaEvent_t start, stop;
    FLOAT time;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    for (int i{0}; i < num_warmups; ++i)
    {
        bound_function(stream);
    }

    cudaStreamSynchronize(stream);

    cudaEventRecord(start, stream);
    for (int i{0}; i < num_repeats; ++i)
    {
        bound_function(stream);
    }
    cudaEventRecord(stop, stream);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&time, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    FLOAT const latency{time / num_repeats};

    return latency;
}