#include <omp.h>
#include <x86intrin.h>
#include <stdio.h>

struct complex {
  float real;
  float imag;
};


void serial_matmul(struct complex ** A, struct complex ** B, struct complex ** C, int a_rows, int a_cols, int b_cols);
void parallel_vectorised_matmul(struct complex ** A, struct complex ** B, struct complex ** C, int a_rows, int a_cols, int b_cols);
void odd_dimension_matmul(struct complex ** A, struct complex ** B, struct complex ** C, int a_rows, int a_cols, int b_cols);
