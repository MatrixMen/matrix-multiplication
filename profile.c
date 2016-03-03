#include <stdlib.h>
#include <sys/time.h>
#include "team_matmul.h"

/* create new empty matrix */
struct complex ** new_empty_matrix(int dim1, int dim2)
{
  struct complex ** result = malloc(sizeof(struct complex*) * dim1);
  struct complex * new_matrix = malloc(sizeof(struct complex) * dim1 * dim2);
  int i;

  for ( i = 0; i < dim1; i++ ) {
    result[i] = &(new_matrix[i*dim2]);
  }

  return result;
}

void free_matrix(struct complex ** matrix) {
  free (matrix[0]); /* free the contents */
  free (matrix); /* free the header */
}

/* take a copy of the matrix and return in a newly allocated matrix */
struct complex ** copy_matrix(struct complex ** source_matrix, int dim1, int dim2)
{
  int i, j;
  struct complex ** result = new_empty_matrix(dim1, dim2);

  for ( i = 0; i < dim1; i++ ) {
    for ( j = 0; j < dim2; j++ ) {
      result[i][j] = source_matrix[i][j];
    }
  }

  return result;
}

/* create a matrix and fill it with random numbers */
struct complex ** gen_random_matrix(int dim1, int dim2)
{
  const int random_range = 512; // constant power of 2
  struct complex ** result;
  int i, j;
  struct timeval seedtime;
  int seed;

  result = new_empty_matrix(dim1, dim2);

  /* use the microsecond part of the current time as a pseudorandom seed */
  gettimeofday(&seedtime, NULL);
  seed = seedtime.tv_usec;
  srandom(seed);

  /* fill the matrix with random numbers */
  for ( i = 0; i < dim1; i++ ) {
    for ( j = 0; j < dim2; j++ ) {
      /* evenly generate values in the range [0, random_range-1)*/
      result[i][j].real = (float)(random() % random_range);
      result[i][j].imag = (float)(random() % random_range);

      /* at no loss of precision, negate the values sometimes */
      /* so the range is now (-(random_range-1), random_range-1)*/
      if (random() & 1) result[i][j].real = -result[i][j].real;
      if (random() & 1) result[i][j].imag = -result[i][j].imag;
    }
  }

  return result;
}

int main(int argc, char** argv) {
  struct complex ** A, ** B, ** C;
  int small = 100;
  int big_odd = 175;
  int big_even = 172;

  A = gen_random_matrix(small, small);
  B = gen_random_matrix(small, small);
  C = new_empty_matrix(small, small);

  serial_matmul(A, B, C, small, small, small);

  A = gen_random_matrix(big_odd, big_odd);
  B = gen_random_matrix(big_odd, big_odd);
  C = new_empty_matrix(big_odd, big_odd);

  odd_dimension_matmul(A, B, C, big_odd, big_odd, big_odd);

  A = gen_random_matrix(big_even, big_even);
  B = gen_random_matrix(big_even, big_even);
  C = new_empty_matrix(big_even, big_even);

  parallel_vectorised_matmul(A, B, C, big_even, big_even, big_even);


  int foo = (int) A[150][150].real;
  printf("WOW: %d\n", foo);

  return 0;
}
