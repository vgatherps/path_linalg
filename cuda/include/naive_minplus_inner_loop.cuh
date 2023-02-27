#pragma once

#include "types.hh"
#include "matrix.cuh"

#include <cuda_runtime.h>

template <bool Branchless>
__device__ inline void naive_inner_loop(int M, int N, int K, int x, int y, int from, int to, const float *A_cost,
                                        const float *B_cost, float &min_cost,
                                        const uint *A_prime, const uint *B_prime,
                                        uint &min_prime)
{
    RowMatrix Am_cost(A_cost, M, K);
    RowMatrix Am_prime(A_prime, M, K);
    RowMatrix Bm_cost(B_cost, K, N);
    RowMatrix Bm_prime(B_prime, K, N);

    for (int i = from; i < to; ++i)
    {

        float i_cost = Am_cost(x, i) + Bm_cost(i, y);
        uint i_prime;

        if (Branchless)
        {
            i_prime = Am_prime(x, i) * Bm_prime(y, i);
        }

        // This is converted into predicates by the compiler
        if (i_cost < min_cost)
        {
            if (!Branchless)
            {
                i_prime = Am_prime(x, i) * Bm_prime(y, i);
            }
            min_cost = i_cost;
            min_prime = i_prime;
        }
    }
}
