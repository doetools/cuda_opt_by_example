#ifndef _DATA_STRUCTURE_H
#define _DATA_STRUCTURE_H

#include <iostream>
#include <vector>
#include <cstdlib>        // random number generator
#include <cuda_runtime.h> // cuda runtime

#define FLOAT float
#define INT int

using namespace std;

template <typename T>
class DataBuffer
{

public:
    const int m; // number of rows
    const int n; // number of columns
    const int size{m * n};
    std::vector<T> c_data;
    T *d_data;

    // add a constructor
    // use the initializer list
    DataBuffer(int num_row = 1, int num_col = 1) : m(num_row), n(num_col)
    {
        // populate the data c
        populate_vector();

        // create device data
        create_device_buffer();

        // copy data to device
        copy_to_device();
    }

    // destructor
    ~DataBuffer()
    {
        // release cuda memory if it is assigned
        if (d_data)
            cudaFree(d_data);
    }

private:
    int populate_vector(int amplifier = 5)
    {
        // seed 100, to avoid change over multiple runs
        srand(100);

        // populate the vector
        for (int i = 0; i < m * n; i++)
        {
            c_data.push_back(T(rand() * amplifier) / T(RAND_MAX));
        }

        return 0;
    }

    int create_device_buffer()
    {
        cudaMalloc(&d_data, size * sizeof(T));
        return 0;
    }

    int copy_to_device()
    {
        cudaMemcpy(d_data, c_data.data(), size * sizeof(T), cudaMemcpyHostToDevice);
        return 0;
    }

    int copy_to_host()
    {
        cudaMemcpy(c_data.data(), d_data, size * sizeof(T), cudaMemcpyDeviceToHost);
        return 0;
    }
};

#endif