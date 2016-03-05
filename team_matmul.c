#include "team_matmul.h"

void serial_matmul(struct complex ** A, struct complex ** B, struct complex ** C, int a_rows, int a_cols, int b_cols) {
  for (int i = 0; i < a_rows; i++) {
    for (int k = 0; k < a_cols; k++) {
      struct complex r = A[i][k];

      for (int j = 0; j < b_cols; j++) {
        struct complex x = B[k][j];
        float real = r.real * x.real - r.imag * x.imag;
        float imag = r.real * x.imag + r.imag * x.real;
        C[i][j].real += real;
        C[i][j].imag += imag;
      }
    }
  }
}

void parallel_vectorised_matmul(struct complex ** A, struct complex ** B, struct complex ** C, int a_rows, int a_cols, int b_cols) {
  #pragma omp parallel for
  for (int i = 0; i < a_rows; i++) {
    for (int k = 0; k < a_cols; k++) {
      struct complex r = A[i][k];

      __m128 a_real = _mm_set1_ps(r.real);
      __m128 a_imag = _mm_set1_ps(r.imag);

      for (int j = 0; j < b_cols; j += 2) {
        __m128 b_complex = _mm_load_ps((float*) &B[k][j]);

        __m128 real_times_b = _mm_mul_ps(a_real, b_complex);
        __m128 imag_times_b = _mm_mul_ps(a_imag, b_complex);
        imag_times_b = _mm_shuffle_ps(imag_times_b, imag_times_b, _MM_SHUFFLE(2, 3, 0, 1));

        __m128 addsub = _mm_addsub_ps(real_times_b, imag_times_b);

        __m128 current_c = _mm_load_ps((float*) &C[i][j]);
        _mm_store_ps((float*) &C[i][j], _mm_add_ps(current_c, addsub));
      }
    }
  }
}

void odd_dimension_matmul(struct complex ** A, struct complex ** B, struct complex ** C, int a_rows, int a_cols, int b_cols) {
  #pragma omp parallel for
  for (int i = 0; i < a_rows; i++) {
    if (i % 2 == 0) {
      /* printf("TOP HALF\n"); */
      for (int k = 0; k < a_cols; k++) {
        struct complex r = A[i][k];

        __m128 a_real = _mm_set1_ps(r.real);
        __m128 a_imag = _mm_set1_ps(r.imag);
        if (k % 2 == 0) {
          for (int j = 0; j < b_cols; j += 2) {
            if(j == b_cols -1) {
              /* printf("Even i, k -- last j\n"); */
              struct complex x = B[k][j];
              float real = r.real * x.real - r.imag * x.imag;
              float imag = r.real * x.imag + r.imag * x.real;
              C[i][j].real += real;
              C[i][j].imag += imag;
            } else {
              /* printf("Even i,k -- other j\n"); */
              __m128 b_complex = _mm_load_ps((float*) &B[k][j]);

              __m128 real_times_b = _mm_mul_ps(a_real, b_complex);
              __m128 imag_times_b = _mm_mul_ps(a_imag, b_complex);
              imag_times_b = _mm_shuffle_ps(imag_times_b, imag_times_b, _MM_SHUFFLE(2, 3, 0, 1));

              __m128 addsub = _mm_addsub_ps(real_times_b, imag_times_b);

              __m128 current_c = _mm_load_ps((float*) &C[i][j]);
              _mm_store_ps((float*) &C[i][j], _mm_add_ps(current_c, addsub));
            }
          }
        } else {
          for (int j = 0; j < b_cols; j += 2) {
            if ( j == b_cols - 1 ) {
              struct complex x = B[k][j];
              float real = r.real * x.real - r.imag * x.imag;
              float imag = r.real * x.imag + r.imag * x.real;
              C[i][j].real += real;
              C[i][j].imag += imag;
            } else {
              __m128 b_complex = _mm_loadu_ps((float*) &B[k][j]);

              __m128 real_times_b = _mm_mul_ps(a_real, b_complex);
              __m128 imag_times_b = _mm_mul_ps(a_imag, b_complex);
              imag_times_b = _mm_shuffle_ps(imag_times_b, imag_times_b, _MM_SHUFFLE(2, 3, 0, 1));

              __m128 addsub = _mm_addsub_ps(real_times_b, imag_times_b);

              __m128 current_c = _mm_loadu_ps((float*) &C[i][j]);
              _mm_storeu_ps((float*) &C[i][j], _mm_add_ps(current_c, addsub));
            }
          }
        }
      }
    } else {
      /* printf("BOTTOM HALF\n"); */
      for (int k = 0; k < a_cols; k++) {
        struct complex r = A[i][k];

        __m128 a_real = _mm_set1_ps(r.real);
        __m128 a_imag = _mm_set1_ps(r.imag);
        if(k % 2 == 0) {
          for(int j = 0; j < b_cols; j+=2) {
            if ( j == b_cols - 1 ) {
              struct complex x = B[k][j];
              float real = r.real * x.real - r.imag * x.imag;
              float imag = r.real * x.imag + r.imag * x.real;
              C[i][j].real += real;
              C[i][j].imag += imag;
            } else {
              __m128 b_complex = _mm_loadu_ps((float*) &B[k][j]);

              __m128 real_times_b = _mm_mul_ps(a_real, b_complex);
              __m128 imag_times_b = _mm_mul_ps(a_imag, b_complex);
              imag_times_b = _mm_shuffle_ps(imag_times_b, imag_times_b, _MM_SHUFFLE(2, 3, 0, 1));

              __m128 addsub = _mm_addsub_ps(real_times_b, imag_times_b);

              __m128 current_c = _mm_loadu_ps((float*) &C[i][j]);
              _mm_storeu_ps((float*) &C[i][j], _mm_add_ps(current_c, addsub));
            }
          }
        } else {
          for(int j = 0; j < b_cols; j+=2) {
            if ( j == 0) {
              struct complex x = B[k][j];
              float real = r.real * x.real - r.imag * x.imag;
              float imag = r.real * x.imag + r.imag * x.real;
              C[i][j].real += real;
              C[i][j].imag += imag;
              j--;
            } else {
              __m128 b_complex = _mm_load_ps((float*) &B[k][j]);

              __m128 real_times_b = _mm_mul_ps(a_real, b_complex);
              __m128 imag_times_b = _mm_mul_ps(a_imag, b_complex);
              imag_times_b = _mm_shuffle_ps(imag_times_b, imag_times_b, _MM_SHUFFLE(2, 3, 0, 1));

              __m128 addsub = _mm_addsub_ps(real_times_b, imag_times_b);

              __m128 current_c = _mm_load_ps((float*) &C[i][j]);
              _mm_store_ps((float*) &C[i][j], _mm_add_ps(current_c, addsub));
            }
          }
        }
      }
    }
  }
}

/* void odd_dimension_matmul(struct complex ** A, struct complex ** B, struct complex ** C, int a_rows, int a_cols, int b_cols) { */
/*   #pragma omp parallel for */
/*   for (int i = 0; i < a_rows; i++) { */
/*     for (int k = 0; k < a_cols; k++) { */
/*       struct complex r = A[i][k]; */

/*       __m128 a_real = _mm_set1_ps(r.real); */
/*       __m128 a_imag = _mm_set1_ps(r.imag); */

/*       for (int j = 0; j < b_cols; j += 2) { */
/*         if ( j == b_cols - 1 ) { */
/*           struct complex x = B[k][j]; */
/*           float real = r.real * x.real - r.imag * x.imag; */
/*           float imag = r.real * x.imag + r.imag * x.real; */
/*           C[i][j].real += real; */
/*           C[i][j].imag += imag; */
/*         } else { */
/*           __m128 b_complex = _mm_loadu_ps((float*) &B[k][j]); */

/*           __m128 real_times_b = _mm_mul_ps(a_real, b_complex); */
/*           __m128 imag_times_b = _mm_mul_ps(a_imag, b_complex); */
/*           imag_times_b = _mm_shuffle_ps(imag_times_b, imag_times_b, _MM_SHUFFLE(2, 3, 0, 1)); */

/*           __m128 add = _mm_add_ps(real_times_b, imag_times_b); */
/*           __m128 sub = _mm_sub_ps(real_times_b, imag_times_b); */

/*           __m128 blender = _mm_blend_ps(sub, add, 10); */

/*           __m128 current_c = _mm_loadu_ps((float*) &C[i][j]); */
/*           _mm_storeu_ps((float*) &C[i][j], _mm_add_ps(current_c, blender)); */
/*         } */
/*       } */
/*     } */
/*   } */
/* } */
