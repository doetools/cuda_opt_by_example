#include "data_structure.h"

template <typename T>
vector<T> matrix_multiplication(vector<T> const &a, vector<T> const &b,
                                const int M, const int N)
{

    vector<T> c(M * M, T(0));

    T total{T(0)};

    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < M; j++)
        {
            total = T(0);
            for (int k = 0; k < N; k++)
            {
                total += a[i * N + k] * b[k * M + j];
            }
            c[i * M + j] = total;
        }
    }

    return c;
}

template <class T>
T max_difference(vector<T> &target, vector<T> &src)
{

    vector<float> diff(target.size(), T(0.0));

    for (int i = 0; i < target.size(); i++)
    {
        diff[i] = fabs(target[i] - src[i]);
    }

    // max_element returns a pointer.
    T max_diff = *max_element(diff.begin(), diff.end());

    return max_diff;
}