#pragma once

#include "host_minplus.hh"
#include "naive_minplus.cuh"
#include "naive_minplus_branch.cuh"
#include "naive_minplus_contiguous.cuh"
#include "naive_minplus_shared.cuh"
#include "types.hh"

#include <cuda_runtime.h>

#include <cstdlib>

typedef void (*KernelFn)(int M, int N, int K, const float *A_cost,
                         const float *B_cost, float *C_cost,
                         const uint *A_prime, const uint *B_prime,
                         uint *C_prime);

struct Kernel {
  const char *name;
  KernelFn fnc;
};

constexpr static Kernel kernels[] = {
    {"host", host_minplus},
    {
        "naive_minplus",
        naive_minplus,
    },
    {
        "naive_minplus_branch",
        naive_minplus_branch,
    },
    {
        "naive_minplus_contiguous",
        naive_minplus_contiguous,
    },
    {"naive_minplus_shared", naive_minplus_shared}};

constexpr static std::size_t num_kernels = sizeof(kernels) / sizeof(Kernel);