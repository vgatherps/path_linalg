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

// Instead of selecting multiple contiguous rows operating on a single column
// we select a single row operating on contiguous columns
// This means for the row-major contiguous operations each thread
// shares the same load, and for row-major column access,
// each thread is acting on contiguous columns of a single row
// and can coalesce the memory accesses

// everything else is the same - we've just changed thread row/column grouping
template <int BLOCK_SIZE>
__global__ static void naive_minplus_contiguous_cu(int M, int N, int K,
                                                   const float *A_cost, const float *B_cost, float *C_cost,
                                                   const uint *A_prime, const uint *B_prime, uint *C_prime)
{
    const uint x = blockIdx.x * BLOCK_SIZE + (threadIdx.x / BLOCK_SIZE);
    const uint y = blockIdx.y * BLOCK_SIZE + (threadIdx.x % BLOCK_SIZE);

    if (K <= 0)
    {
        return;
    }

    if (x < M && y < N)
    {
        float min_cost = A_cost[x * K] + B_cost[y];
        uint min_prime = A_prime[x * K] * B_prime[y];
        int min_k_branchless = K < NUM_ROWS_BRANCHLESS ? K : K;
        naive_inner_loop<true>(M, N, K, x, y, 1, min_k_branchless, A_cost, B_cost, min_cost, A_prime, B_prime, min_prime);
        naive_inner_loop<false>(M, N, K, x, y, min_k_branchless, K, A_cost, B_cost, min_cost, A_prime, B_prime, min_prime);
        C_cost[x * N + y] = min_cost;
        C_prime[x * N + y] = min_prime;
    }
}

void naive_minplus_contiguous(int M, int N, int K,
                              const float *A_cost, const float *B_cost, float *C_cost,
                              const uint *A_prime, const uint *B_prime, uint *C_prime)
{
    dim3 gridDim(CEIL_DIV(M, 32), CEIL_DIV(N, 32));
    dim3 blockDim(32 * 32);
    naive_minplus_contiguous_cu<32><<<gridDim, blockDim>>>(M, N, K, A_cost, B_cost, C_cost, A_prime, B_prime, C_prime);
}