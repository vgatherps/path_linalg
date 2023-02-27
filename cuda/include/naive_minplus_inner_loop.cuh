#pragma once

#include "types.hh"

#include <cuda_runtime.h>

template <bool Branchless>
__device__ inline void naive_inner_loop(int M, int N, int K, int x, int y, int from, int to, const float *A_cost,
                                        const float *B_cost, float &min_cost,
                                        const uint *A_prime, const uint *B_prime,
                                        uint &min_prime)
{
    for (int i = from; i < to; ++i)
    {

        float i_cost = A_cost[x * K + i] + B_cost[i * N + y];
        uint i_prime;

        if (Branchless)
        {
            i_prime = A_prime[x * K + i] * B_prime[i * N + y];
        }

        // This is converted into predicates by the compiler
        if (i_cost < min_cost)
        {
            if (!Branchless)
            {
                i_prime = A_prime[x * K + i] * B_prime[i * N + y];
            }
            min_cost = i_cost;
            min_prime = i_prime;
        }
    }
}
