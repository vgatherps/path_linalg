#include "types.hh"

#include "naive_minplus.cuh"

typedef __global__ void (*KernelFn)(int M, int N, int K,
                                    const float *A_cost, const float *B_cost, float *C_cost,
                                    const uint *A_prime, const uint *B_prime, uint *C_prime);

struct Kernel
{
    const char *name;
    KernelFn fnc;
};

constexpr Kernel kernels[] = {
    {"naive_minplus", naive_minplus},
};