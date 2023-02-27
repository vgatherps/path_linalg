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
__global__ static void naive_minplus_shared_cu(int M, int N, int K,
                                               const float *A_cost, const float *B_cost, float *C_cost,
                                               const uint *A_prime, const uint *B_prime, uint *C_prime)
{

    const uint block_x = blockIdx.x;
    const uint block_y = blockIdx.y;

    const uint thread_x = threadIdx.x / BLOCK_SIZE;
    const uint thread_y = threadIdx.x % BLOCK_SIZE;

    // allocate buffer for current block in fast shared mem
    // shared mem is shared between all threads in a block

    // double buffer to reduce synchronization
    __shared__ float A_cost_shared[2][BLOCK_SIZE * BLOCK_SIZE];
    __shared__ float B_cost_shared[2][BLOCK_SIZE * BLOCK_SIZE];
    __shared__ uint A_prime_shared[2][BLOCK_SIZE * BLOCK_SIZE];
    __shared__ uint B_prime_shared[2][BLOCK_SIZE * BLOCK_SIZE];

    if (K <= 0)
    {
        return;
    }

    // This is somewhat incomplete as we don't compute on the edges
    // our inputs are always rounded so this kindof works

    A_cost += block_x * BLOCK_SIZE * K;
    A_prime += block_x * BLOCK_SIZE * K;
    B_cost += block_y * BLOCK_SIZE;
    B_prime += block_y * BLOCK_SIZE;
    C_cost += block_x * BLOCK_SIZE * N + block_y * BLOCK_SIZE;
    C_prime += block_x * BLOCK_SIZE * N + block_y * BLOCK_SIZE;

    // minor inefficiency here - we aggressively load the first
    // and then do min-plus along the whole matrix
    // It only results in a very minor amount of extra work
    // could be avoided to almost no upside

    float min_cost = A_cost[thread_x * K] + B_cost[thread_y];
    uint min_prime = A_prime[thread_x * K] * B_prime[thread_y];

    int write_block = 0;

    A_cost_shared[0][thread_x * BLOCK_SIZE + thread_y] = A_cost[thread_x * K + thread_y];
    A_prime_shared[0][thread_x * BLOCK_SIZE + thread_y] = A_prime[thread_x * K + thread_y];
    B_cost_shared[0][thread_x * BLOCK_SIZE + thread_y] = B_cost[thread_x * N + thread_y];
    B_prime_shared[0][thread_x * BLOCK_SIZE + thread_y] = B_prime[thread_x * N + thread_y];
    __syncthreads();

    for (int whichBlock = 0; whichBlock < K; whichBlock += BLOCK_SIZE)
    {
        int read_block = write_block;
        write_block = 1 - write_block;

        A_cost += BLOCK_SIZE;
        A_prime += BLOCK_SIZE;
        B_cost += BLOCK_SIZE * N;
        B_prime += BLOCK_SIZE * N;

        // each thread loads a single element into shared memory
        A_cost_shared[write_block][thread_x * BLOCK_SIZE + thread_y] = A_cost[thread_x * K + thread_y];
        A_prime_shared[write_block][thread_x * BLOCK_SIZE + thread_y] = A_prime[thread_x * K + thread_y];
        B_cost_shared[write_block][thread_x * BLOCK_SIZE + thread_y] = B_cost[thread_x * N + thread_y];
        B_prime_shared[write_block][thread_x * BLOCK_SIZE + thread_y] = B_prime[thread_x * N + thread_y];

        // do min-plus on the block
        naive_inner_loop<true>(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE, thread_x, thread_y, 0, BLOCK_SIZE, A_cost_shared[read_block], B_cost_shared[read_block], min_cost, A_prime_shared[read_block], B_prime_shared[read_block], min_prime);

        __syncthreads(); // block to ensure everyone has read into write_block and finished reading from read_block
    }
    C_cost[thread_x * N + thread_y] = min_cost;
    C_prime[thread_x * N + thread_y] = min_prime;
}

void naive_minplus_shared(int M, int N, int K,
                          const float *A_cost, const float *B_cost, float *C_cost,
                          const uint *A_prime, const uint *B_prime, uint *C_prime)
{
    dim3 gridDim(CEIL_DIV(M, 32), CEIL_DIV(N, 32));
    dim3 blockDim(32 * 32);
    cudaFuncSetAttribute(naive_minplus_shared_cu<32>,
                         cudaFuncAttributePreferredSharedMemoryCarveout,
                         cudaSharedmemCarveoutMaxShared);
    naive_minplus_shared_cu<32><<<gridDim, blockDim>>>(M, N, K, A_cost, B_cost, C_cost, A_prime, B_prime, C_prime);
}