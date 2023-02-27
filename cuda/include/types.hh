#pragma once

using uint = unsigned int;

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

#define ROW_IDX(M, Rs, Cs, r, c) (M)[(r) * (Cs) + (c)]
