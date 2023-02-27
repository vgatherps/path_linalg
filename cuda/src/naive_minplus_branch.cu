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

constexpr static int NUM_ROWS_BRANCHLESS = 50;

// For the first 50 rows of the matrix, go branchless as we assume there will be many overwrites
// however assume that by 50 rows, we'll have enough points to be close th a true minimum
// and can skip over the rest
__global__ static void naive_minplus_branch_cu(int M, int N, int K,
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
        int min_k_branchless = K < NUM_ROWS_BRANCHLESS ? K : NUM_ROWS_BRANCHLESS;
        naive_inner_loop<true>(M, N, K, x, y, 1, min_k_branchless, A_cost, B_cost, min_cost, A_prime, B_prime, min_prime);
        naive_inner_loop<false>(M, N, K, x, y, min_k_branchless, K, A_cost, B_cost, min_cost, A_prime, B_prime, min_prime);
        C_cost[x * N + y] = min_cost;
        C_prime[x * N + y] = min_prime;
    }
}

void naive_minplus_branch(int M, int N, int K,
                          const float *A_cost, const float *B_cost, float *C_cost,
                          const uint *A_prime, const uint *B_prime, uint *C_prime)
{
    dim3 gridDim(CEIL_DIV(M, 32), CEIL_DIV(N, 32));
    dim3 blockDim(32, 32);
    naive_minplus_branch_cu<<<gridDim, blockDim>>>(M, N, K, A_cost, B_cost, C_cost, A_prime, B_prime, C_prime);
}