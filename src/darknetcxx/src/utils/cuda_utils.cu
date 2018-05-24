#include "darknetcxx/cuda_utils.hpp"
#include <cassert>
#include <cstdio>

namespace darknet {
#ifdef DARKNET_MALLOCATED
template<>
double Ptr_gpu<float>::m_allocated = 0;
template<>
double Ptr_gpu<int>::m_allocated = 0;
template<>
double Ptr_gpu<unsigned char>::m_allocated = 0;
#endif

int gpu_index = 0;

cublasHandle_t
blas_handle()
{
    static int init[16] = { 0 };
    static cublasHandle_t handle[16];
    int i = cuda_get_device();
    if (!init[i]) {
        cublasCreate(&handle[i]);
        init[i] = 1;
    }
    return handle[i];
}

dim3
cuda_gridsize(size_t n)
{
    assert(n > 0);
    unsigned int k = n / BLOCKSIZE;
    if (n % BLOCKSIZE != 0)
        k += 1;

    unsigned int x = k;
    unsigned int y = 1;
    if (x > 65535) {
        x = ceil(sqrt(k));
        y = n / (x * BLOCKSIZE);
        if (n % (x * BLOCKSIZE) != 0)
            y += 1;
    }
    dim3 d = { x, y, 1 };
    return d;
}

void
cuda_set_device(int n)
{
    gpu_index = n;
    checkCudaErrors(cudaSetDevice(n));
}

int
cuda_get_device()
{
    int n = 0;
    checkCudaErrors(cudaGetDevice(&n));
    return n;
}

cudnnHandle_t
cudnn_handle()
{
    static int init[16] = { 0 };
    static cudnnHandle_t handle[16];
    int i = cuda_get_device();
    if (!init[i]) {
        cudnnCreate(&handle[i]);
        init[i] = 1;
    }
    return handle[i];
}

} // namespace darknet
