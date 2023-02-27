#pragma once

#include "types.hh"
#include <cuda_runtime.h>

template <class T>
struct RowMatrix
{
    T *data;
    int rows;
    int cols;

    __device__ RowMatrix(T *d, int r, int c) : data(d), rows(r), cols(c) {}

    __device__ inline T &operator()(int r, int c)
    {
        return ROW_IDX(data, rows, cols, r, c);
    }
};