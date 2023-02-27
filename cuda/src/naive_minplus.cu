#include "types.hh"

#include "naive_minplus_inner_loop.cuh"

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

/*
Matrix sizes:
MxK * KxN = MxN
*/

// We assume that A, B, and C are stored in row-major order

__global__ static void naive_minplus_cu(int M, int N, int K,
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

        naive_inner_loop<true>(M, N, K, x, y, 1, K, A_cost, B_cost, min_cost, A_prime, B_prime, min_prime);

        C_cost[x * N + y] = min_cost;
        C_prime[x * N + y] = min_prime;
    }
}

void naive_minplus(int M, int N, int K,
                   const float *A_cost, const float *B_cost, float *C_cost,
                   const uint *A_prime, const uint *B_prime, uint *C_prime)
{
    dim3 gridDim(CEIL_DIV(M, 32), CEIL_DIV(N, 32));
    dim3 blockDim(32, 32);
    naive_minplus_cu<<<gridDim, blockDim>>>(M, N, K, A_cost, B_cost, C_cost, A_prime, B_prime, C_prime);
}