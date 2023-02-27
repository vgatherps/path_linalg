#include "host_minplus.hh"
#include "kernels.cuh"
#include "runner.hh"

#include <cuda_runtime.h>

#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fstream>
#include <iostream>
#include <vector>

#ifndef ALWAYS_VERIFY
#define ALWAYS_VERIFY false
#endif

#define cudaCheck(err) (cudaCheck(err, __FILE__, __LINE__))

const std::string errLogFile = "matrixValidationFailure.txt";

// Largely taken from the sgemm example

int main(int argc, char **argv) {
  if (argc != 2) {
    std::cerr << "Please select a kernel (range 0 - " << num_kernels
              << ", 0 for naive CPU solution)" << std::endl;
    exit(EXIT_FAILURE);
  }

  // get kernel number
  int kernel_num = std::stoi(argv[1]);
  if (kernel_num < 0 || kernel_num > num_kernels) {
    std::cerr << "Please enter a valid kernel number (0-" << num_kernels << ")"
              << std::endl;
    exit(EXIT_FAILURE);
  }

  // get environment variable for device
  int deviceIdx = 0;
  if (getenv("DEVICE") != NULL) {
    deviceIdx = atoi(getenv("DEVICE"));
  }
  cudaSetDevice(deviceIdx);

  CudaDeviceInfo();

  Kernel kernel = kernels[kernel_num];

  std::cout << "Running kernel " << kernel.name << " number " << kernel_num
            << " on device " << deviceIdx << std::endl;

  // Using cudaEvent for gpu stream timing, cudaEvent is equivalent to
  // publishing event tasks in the target stream
  float elapsed_time;
  cudaEvent_t beg, end;
  cudaEventCreate(&beg);
  cudaEventCreate(&end);

  // cuBLAS FLOPs ceiling is reached at 8192
  std::vector<int> SIZE = {128, 256, 512, 1024, 2048, 4096};

  long m, n, k, max_size;
  max_size = SIZE[SIZE.size() - 1];
  std::cout << "Max size: " << max_size << std::endl;

  float alpha = 0.5, beta = 3.0; // GEMM input parameters, C=α*AB+β*C

  float *A_cost = nullptr, *B_cost = nullptr, *C_cost, *C_ref_cost,
        *C_host_cost = nullptr; // host cost matrices
  uint *A_prime = nullptr, *B_prime = nullptr, *C_prime, *C_ref_prime = nullptr,
       *C_host_prime; // host prime matrices
  float *dA_cost = nullptr, *dB_cost = nullptr,
        *dC_cost = nullptr; // device cost matrices
  uint *dA_prime = nullptr, *dB_prime = nullptr,
       *dC_prime = nullptr; // device prime matrices

  std::size_t matrix_size = sizeof(float) * max_size * max_size;
  static_assert(sizeof(uint) ==
                sizeof(float)); // ensure we can reuse the matrix size

  A_cost = (float *)malloc(matrix_size);
  B_cost = (float *)malloc(matrix_size);
  C_cost = (float *)malloc(matrix_size);
  C_ref_cost = (float *)malloc(matrix_size);
  C_host_cost = (float *)malloc(matrix_size);
  A_prime = (uint *)malloc(matrix_size);
  B_prime = (uint *)malloc(matrix_size);
  C_prime = (uint *)malloc(matrix_size);
  C_ref_prime = (uint *)malloc(matrix_size);
  C_host_prime = (uint *)malloc(matrix_size);

  randomize_matrix(A_cost, max_size * max_size);
  randomize_matrix(B_cost, max_size * max_size);
  randomize_matrix(C_cost, max_size * max_size);
  randomize_uint_matrix(A_prime, max_size * max_size);
  randomize_uint_matrix(B_prime, max_size * max_size);
  randomize_uint_matrix(C_prime, max_size * max_size);

  std::memcpy(C_ref_cost, C_cost, matrix_size);
  std::memcpy(C_ref_prime, C_prime, matrix_size);

  cudaCheck(cudaMalloc((void **)&dA_cost, matrix_size));
  cudaCheck(cudaMalloc((void **)&dB_cost, matrix_size));
  cudaCheck(cudaMalloc((void **)&dC_cost, matrix_size));

  cudaCheck(cudaMalloc((void **)&dA_prime, matrix_size));
  cudaCheck(cudaMalloc((void **)&dB_prime, matrix_size));
  cudaCheck(cudaMalloc((void **)&dC_prime, matrix_size));

  cudaCheck(cudaMemcpy(dA_cost, A_cost, matrix_size, cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpy(dB_cost, B_cost, matrix_size, cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpy(dC_cost, C_cost, matrix_size, cudaMemcpyHostToDevice));

  cudaCheck(cudaMemcpy(dA_prime, A_prime, matrix_size, cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpy(dB_prime, B_prime, matrix_size, cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpy(dC_prime, C_prime, matrix_size, cudaMemcpyHostToDevice));

  int repeat_times = 50;

  for (int size : SIZE) {
    m = n = k = size;

    int local_matrix_size = sizeof(float) * n * m;

    std::memcpy(C_cost, C_ref_cost, local_matrix_size);
    std::memcpy(C_prime, C_ref_prime, local_matrix_size);

    cudaCheck(
        cudaMemcpy(dC_cost, C_cost, local_matrix_size, cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(dC_prime, C_prime, local_matrix_size,
                         cudaMemcpyHostToDevice));

    std::cout << "dimensions(m=n=k) " << m << std::endl;

    // Verify the correctness of the calculation, and execute it once before the
    // kernel function timing to avoid cold start errors
    if (kernel_num != 0) {
      kernel.fnc(m, n, k, dA_cost, dB_cost, dC_cost, dA_prime, dB_prime,
                 dC_prime);

      cudaCheck(cudaDeviceSynchronize());
      cudaMemcpy(C_cost, dC_cost, local_matrix_size, cudaMemcpyDeviceToHost);
      cudaMemcpy(C_prime, dC_prime, local_matrix_size, cudaMemcpyDeviceToHost);
      if (ALWAYS_VERIFY || size < 512) {

        host_minplus(m, n, k, A_cost, B_cost, C_host_cost, A_prime, B_prime,
                     C_host_prime);
        if (!verify_matrix(C_cost, C_host_cost, m * n)) {
          std::cout << "Failed to pass the correctness verification against "
                       "host impl "
                    << std::endl;
          /*
   if (m <= 128) {
     std::cout << " Logging faulty output into " << errLogFile << "\n";
     std::ofstream fs;
     fs.open(errLogFile);
     fs << "A:\n";
     print_matrix(A, m, n, fs);
     fs << "B:\n";
     print_matrix(B, m, n, fs);
     fs << "C:\n";
     print_matrix(C, m, n, fs);
     fs << "Should:\n";
     print_matrix(C_ref, m, n, fs);
   }
   */
          exit(EXIT_FAILURE);
        }
      }
    }

    cudaEventRecord(beg);
    for (int j = 0; j < repeat_times; j++) {
      if (kernel_num != 0) {

        kernel.fnc(m, n, k, dA_cost, dB_cost, dC_cost, dA_prime, dB_prime,
                   dC_prime);
      } else {
        kernel.fnc(m, n, k, A_cost, B_cost, C_cost, A_prime, B_prime, C_prime);
      }
    }
    cudaEventRecord(end);
    cudaEventSynchronize(beg);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&elapsed_time, beg, end);
    elapsed_time /= 1000.; // Convert to seconds

    long flops = 2 * m * n * k;
    std::cout << "Elapsed time: " << elapsed_time << " s, performance: "
              << (repeat_times * flops * 1e-9) / elapsed_time << " GFLOPS."
              << std::endl;
  }

  free(A_cost);
  free(B_cost);
  free(C_cost);
  free(C_ref_cost);
  cudaFree(dA_cost);
  cudaFree(dB_cost);
  cudaFree(dC_cost);

  free(A_prime);
  free(B_prime);
  free(C_prime);
  free(C_ref_prime);
  cudaFree(dA_prime);
  cudaFree(dB_prime);
  cudaFree(dC_prime);

  return 0;
};