#pragma once

#include "types.hh"

#include <cuda_runtime.h>

__global__ void naive_minplus_branch(int M, int N, int K, const float *A_cost,
                                     const float *B_cost, float *C_cost,
                                     const uint *A_prime, const uint *B_prime,
                                     uint *C_prime);