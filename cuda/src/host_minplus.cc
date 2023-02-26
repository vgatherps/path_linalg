#include "types.hh"

/*
Matrix sizes:
MxK * KxN = MxN
*/

// We assume that A, B, and C are stored in row-major order

void host_minplus(int M, int N, int K, const float *A_cost, const float *B_cost,
                  float *C_cost, const uint *A_prime, const uint *B_prime,
                  uint *C_prime) {

  if (K <= 0) {
    return;
  }
  for (int x = 0; x < M; x++) {
    for (int y = 0; y < N; y++) {
      float min_cost = A_cost[x * K] + B_cost[y];
      uint min_prime = A_prime[x * K] * B_prime[y];
      for (int i = 1; i < K; ++i) {
        float i_cost = A_cost[x * K + i] + B_cost[y];
        if (i_cost < min_cost) {
          min_cost = i_cost;
          min_prime = A_prime[x * K + i] * B_prime[i * N + y];
        }
      }
      C_cost[x * N + y] = min_cost;
      C_prime[x * N + y] = min_prime;
    }
  }
}