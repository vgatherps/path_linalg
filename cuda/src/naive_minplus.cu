#include "types.hh"

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

/*
Matrix sizes:
MxK * KxN = MxN
*/

// We assume that A, B, and C are stored in row-major order

__global__ void naive_minplus(int M, int N, int K,
                              const float *A_cost, const float *B_cost, float *C_cost,
                              const uint *A_prime, const uint *B_prime, uint *C_prime)
{
    const uint x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint y = blockIdx.y * blockDim.y + threadIdx.y;

    if (K <= 0)
    {
        return;
    }

    if (x < M && y < N)
    {
        float min_cost = A_cost[x * K] + B_cost[y];
        uint min_prime = A_prime[x * K] * B_prime[y];
        for (int i = 1; i < K; ++i)
        {

            float i_cost = A_cost[x * K + i] + B_cost[y];
            uint i_prime = A_prime[x * K + i] * B_prime[i * N + y];

            // This is converted into predicates by the compiler
            if (i_cost < min_cost)
            {
                min_cost = i_cost;
                min_prime = i_prime;
            }
        }
        C_cost[x * N + y] = min_cost;
        C_prime[x * N + y] = min_prime;
    }
}