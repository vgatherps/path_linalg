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

    RowMatrix<const float> Am_cost(A_cost, M, K);
    RowMatrix<const uint> Am_prime(A_prime, M, K);
    RowMatrix<const float> Bm_cost(B_cost, K, N);
    RowMatrix<const uint> Bm_prime(B_prime, K, N);
    RowMatrix<float> Cm_cost(C_cost, M, N);
    RowMatrix<uint> Cm_prime(C_prime, M, N);

    // allocate buffer for current block in fast shared mem
    // shared mem is shared between all threads in a block

    // double buffer to reduce synchronization
    __shared__ float A_cost_shared[2][BLOCK_SIZE * BLOCK_SIZE];
    __shared__ float B_cost_shared[2][BLOCK_SIZE * BLOCK_SIZE];
    __shared__ uint A_prime_shared[2][BLOCK_SIZE * BLOCK_SIZE];
    __shared__ uint B_prime_shared[2][BLOCK_SIZE * BLOCK_SIZE];

    RowMatrix<float> Am_cost_shared[] = {RowMatrix(A_cost_shared[0], BLOCK_SIZE, BLOCK_SIZE),
                                         RowMatrix(A_cost_shared[1], BLOCK_SIZE, BLOCK_SIZE)};
    RowMatrix<uint> Am_prime_shared[] = {RowMatrix(A_prime_shared[0], BLOCK_SIZE, BLOCK_SIZE),
                                         RowMatrix(A_prime_shared[1], BLOCK_SIZE, BLOCK_SIZE)};
    RowMatrix<float> Bm_cost_shared[] = {RowMatrix(B_cost_shared[0], BLOCK_SIZE, BLOCK_SIZE),
                                         RowMatrix(B_cost_shared[1], BLOCK_SIZE, BLOCK_SIZE)};
    RowMatrix<uint> Bm_prime_shared[] = {RowMatrix(B_prime_shared[0], BLOCK_SIZE, BLOCK_SIZE),
                                         RowMatrix(B_prime_shared[1], BLOCK_SIZE, BLOCK_SIZE)};

    if (K <= 0)
    {
        return;
    }

    // This is somewhat incomplete as we don't compute on the edges
    // our inputs are always rounded so this kindof works

    // C_cost += block_x * BLOCK_SIZE * N + block_y * BLOCK_SIZE;
    // C_prime += block_x * BLOCK_SIZE * N + block_y * BLOCK_SIZE;

    const int base_a_row = block_x * BLOCK_SIZE;
    const int base_b_col = block_y * BLOCK_SIZE;

    // minor inefficiency here - we aggressively load the first
    // and then do min-plus along the whole matrix
    // It only results in a very minor amount of extra work
    // could be avoided to almost no upside

    float min_cost = Am_cost(base_a_row + thread_x, 0) + Bm_cost(0, base_b_col + thread_y);
    uint min_prime = Am_prime(base_a_row + thread_x, 0) * Bm_prime(0, base_b_col + thread_y);

    int write_block = 0;

    auto set_writer = [&](int which, int block_offset)
    {
        Am_cost_shared[which](thread_x, thread_y) = Am_cost(base_a_row + thread_x, thread_y + block_offset);
        Am_prime_shared[which](thread_x, thread_y) = Am_prime(base_a_row + thread_x, thread_y + block_offset);
        Bm_cost_shared[which](thread_x, thread_y) = Bm_cost(thread_x + block_offset, base_b_col + thread_y);
        Bm_prime_shared[which](thread_x, thread_y) = Bm_prime(thread_x + block_offset, base_b_col + thread_y);
    };

    set_writer(0, 0);
    __syncthreads();

    for (int whichBlock = 0; whichBlock < K; whichBlock += BLOCK_SIZE)
    {
        int read_block = write_block;
        write_block = 1 - write_block;

        if (whichBlock + BLOCK_SIZE < K)
        {
            set_writer(write_block, whichBlock + BLOCK_SIZE);
        }
        // do min-plus on the block
        naive_inner_loop<true>(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE, thread_x, thread_y, 0, BLOCK_SIZE, &A_cost_shared[read_block][0], &B_cost_shared[read_block][0], min_cost, &A_prime_shared[read_block][0], &B_prime_shared[read_block][0], min_prime);

        __syncthreads(); // block to ensure everyone has read into write_block and finished reading from read_block
    }

    // base row offset
    Cm_cost(base_a_row + thread_x, base_b_col + thread_y) = min_cost;
    Cm_prime(base_a_row + thread_x, base_b_col + thread_y) = min_prime;
}

void naive_minplus_indexing_shared(int M, int N, int K,
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