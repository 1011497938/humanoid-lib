#include "darknetcxx/gemm.hpp"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#ifdef DARKNET_GPU
#include "cuda.h"
#include "darknetcxx/cuda_utils.hpp"
#endif
#include "darknetcxx/utils.hpp"

namespace darknet {

void
gemm_bin(int M, int N, int K, float ALPHA, char* A, int lda, float* B, int ldb, float* C, int ldc)
{
    int i, j, k;
    for (i = 0; i < M; ++i) {
        for (k = 0; k < K; ++k) {
            char A_PART = A[i * lda + k];
            if (A_PART) {
                for (j = 0; j < N; ++j) {
                    C[i * ldc + j] += B[k * ldb + j];
                }
            } else {
                for (j = 0; j < N; ++j) {
                    C[i * ldc + j] -= B[k * ldb + j];
                }
            }
        }
    }
}

float*
random_matrix(int rows, int cols)
{
    int i;
    //  float *m = (float *)calloc(rows * cols, sizeof(float));
    auto m = new float[rows * cols];
    for (i = 0; i < rows * cols; ++i) {
        m[i] = (float)rand() / RAND_MAX;
    }
    return m;
}

void
time_random_matrix(int TA, int TB, int m, int k, int n)
{
    float* a;
    if (!TA)
        a = random_matrix(m, k);
    else
        a = random_matrix(k, m);
    int lda = (!TA) ? k : m;
    float* b;
    if (!TB)
        b = random_matrix(k, n);
    else
        b = random_matrix(n, k);
    int ldb = (!TB) ? n : k;

    float* c = random_matrix(m, n);
    int i;
    clock_t start = clock(), end;
    for (i = 0; i < 10; ++i) {
        gemm_cpu(TA, TB, m, n, k, 1, a, lda, b, ldb, 1, c, n);
    }
    end = clock();
    printf("Matrix Multiplication %dx%d * %dx%d, TA=%d, TB=%d: %lf ms\n", m, k, k, n, TA, TB, (float)(end - start) / CLOCKS_PER_SEC);
    free(a);
    free(b);
    free(c);
}

void
gemm(int TA, int TB, int M, int N, int K, float ALPHA, float* A, int lda, float* B, int ldb, float BETA, float* C, int ldc)
{
    gemm_cpu(TA, TB, M, N, K, ALPHA, A, lda, B, ldb, BETA, C, ldc);
}

void
gemm_nn(int M, int N, int K, float ALPHA, float* A, int lda, float* B, int ldb, float* C, int ldc)
{
    int i, j, k;
    for (i = 0; i < M; ++i) {
        for (k = 0; k < K; ++k) {
            register float A_PART = ALPHA * A[i * lda + k];
            for (j = 0; j < N; ++j) {
                C[i * ldc + j] += A_PART * B[k * ldb + j];
            }
        }
    }
}

void
gemm_nt(int M, int N, int K, float ALPHA, float* A, int lda, float* B, int ldb, float* C, int ldc)
{
    int i, j, k;
    for (i = 0; i < M; ++i) {
        for (j = 0; j < N; ++j) {
            register float sum = 0;
            for (k = 0; k < K; ++k) {
                sum += ALPHA * A[i * lda + k] * B[j * ldb + k];
            }
            C[i * ldc + j] += sum;
        }
    }
}

void
gemm_tn(int M, int N, int K, float ALPHA, float* A, int lda, float* B, int ldb, float* C, int ldc)
{
    int i, j, k;
    for (i = 0; i < M; ++i) {
        for (k = 0; k < K; ++k) {
            register float A_PART = ALPHA * A[k * lda + i];
            for (j = 0; j < N; ++j) {
                C[i * ldc + j] += A_PART * B[k * ldb + j];
            }
        }
    }
}

void
gemm_tt(int M, int N, int K, float ALPHA, float* A, int lda, float* B, int ldb, float* C, int ldc)
{
    int i, j, k;
    for (i = 0; i < M; ++i) {
        for (j = 0; j < N; ++j) {
            register float sum = 0;
            for (k = 0; k < K; ++k) {
                sum += ALPHA * A[i + k * lda] * B[k + j * ldb];
            }
            C[i * ldc + j] += sum;
        }
    }
}

void
gemm_cpu(int TA, int TB, int M, int N, int K, float ALPHA, float* A, int lda, float* B, int ldb, float BETA, float* C, int ldc)
{
    // printf("cpu: %d %d %d %d %d %f %d %d %f %d\n",TA, TB, M, N, K, ALPHA, lda,
    // ldb, BETA, ldc);
    int i, j;
    for (i = 0; i < M; ++i) {
        for (j = 0; j < N; ++j) {
            C[i * ldc + j] *= BETA;
        }
    }
    if (!TA && !TB)
        gemm_nn(M, N, K, ALPHA, A, lda, B, ldb, C, ldc);
    else if (TA && !TB)
        gemm_tn(M, N, K, ALPHA, A, lda, B, ldb, C, ldc);
    else if (!TA && TB)
        gemm_nt(M, N, K, ALPHA, A, lda, B, ldb, C, ldc);
    else
        gemm_tt(M, N, K, ALPHA, A, lda, B, ldb, C, ldc);
}

#ifdef DARKNET_GPU
void
gemm_ongpu(int TA, int TB, int M, int N, int K, float ALPHA, float* A_gpu, int lda, float* B_gpu, int ldb, float BETA, float* C_gpu, int ldc)
{
    cublasHandle_t handle = blas_handle();
    // cublasStatus_t status =
    cublasSgemm(handle, (TB ? CUBLAS_OP_T : CUBLAS_OP_N), (TA ? CUBLAS_OP_T : CUBLAS_OP_N), N, M, K, &ALPHA, B_gpu, ldb, A_gpu, lda, &BETA, C_gpu, ldc);
    //  checkCudaErrors(status);
    checkCudaErrors(cudaPeekAtLastError());
}

void
gemm_gpu(int TA, int TB, int M, int N, int K, float ALPHA, float* A, int lda, float* B, int ldb, float BETA, float* C, int ldc)
{
    float* A_gpu = cuda_make_array(A, (TA ? lda * K : lda * M));
    float* B_gpu = cuda_make_array(B, (TB ? ldb * N : ldb * K));
    float* C_gpu = cuda_make_array(C, ldc * M);

    gemm_ongpu(TA, TB, M, N, K, ALPHA, A_gpu, lda, B_gpu, ldb, BETA, C_gpu, ldc);

    cuda_pull_array(C_gpu, C, ldc * M);
    cuda_free(A_gpu);
    cuda_free(B_gpu);
    cuda_free(C_gpu);
}
#endif

} // namespace darknet
