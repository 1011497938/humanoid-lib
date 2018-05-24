#pragma once
#include "darknetcxx/blas.hpp"
#include "darknetcxx/utils.hpp"
#include <cassert>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cudnn.h>
#include <curand.h>
#include <iostream>
#include <memory>

#ifdef DARKNET_GPU

namespace darknet {
extern int gpu_index;

// #define BLOCKSIZE 1024
constexpr int BLOCKSIZE = 1024; // TODO 512?

cublasHandle_t
blas_handle();

#define checkCudaErrors(val) check((val), #val, __FILE__, __LINE__)

template<typename T>
void
check(T err, const char* const func, const char* const file, const int line)
{
    if (err != cudaSuccess) {
        std::cerr << "CUDA error at: " << file << ":" << line << std::endl;
        std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
        exit(1);
    }
}

template<typename T>
T*
cuda_make_array(T* h_array, size_t numElements)
{
    T* d_array;
    checkCudaErrors(cudaMalloc(&d_array, sizeof(T) * numElements));

    // TODO: fix this
    if (h_array)
        checkCudaErrors(cudaMemcpy(d_array, h_array, sizeof(T) * numElements, cudaMemcpyHostToDevice));
    if (!d_array)
        error("Cuda malloc failed\n");
    return d_array;
}

template<typename T>
void
cuda_free(T* x)
{
    checkCudaErrors(cudaFree(x));
}

template<typename T>
void
cuda_push_array(T* d_array, T* h_array, size_t n)
{
    size_t size = sizeof(T) * n;
    checkCudaErrors(cudaMemcpy(d_array, h_array, size, cudaMemcpyHostToDevice));
}

template<typename T>
void
cuda_pull_array(T* d_array, T* h_array, size_t n)
{
    size_t size = sizeof(T) * n;
    // std::cout << "::size::" << size << std::endl;
    checkCudaErrors(cudaMemcpy(h_array, d_array, size, cudaMemcpyDeviceToHost));
}

dim3
cuda_gridsize(size_t n);

void
cuda_set_device(int n);

int
cuda_get_device();
//  cudaDeviceSynchronize();
//  checkCudaErrors(cudaGetLastError());

cudnnHandle_t
cudnn_handle();

/*******************************************************************************
 * wrapper for raw gpu device_pointers
 ******************************************************************************/

template<typename T>
class Ptr_gpu : NonCopyable
{
  public:
    // template <typename X>
    struct deleter
    {
        void operator()(T* p)
        {
            if (p) {
                cuda_free(p);
            }
        }
    };

    Ptr_gpu()
      : m_size(0)
      , m_ptr(nullptr, deleter())
    {
        // std::cout << "Ptr_gpu ctor" << std::endl;
    }
    Ptr_gpu(int size)
      : m_size(size)
      , m_ptr(cuda_make_array<T>(nullptr, size), deleter())
    {
        // std::cout << "Ptr_gpu ctor" << std::endl;
    }

    T* get()
    {
        return m_ptr.get();
    }

    void resize(int size)
    {
#ifdef DARKNET_MALLOCATED
        m_allocated -= sizeof(T) * m_size / 1000000.;
        m_allocated += sizeof(T) * size / 1000000.;
        //printf("GPU resized from %lu to %lu bytes, all %lf MB\n", sizeof(T) * m_size, sizeof(T) * size, m_allocated);
#endif
        // std::cout << "Ptr gpu resized to " << size << std::endl;
        if (m_ptr.get() != nullptr) {
            cuda_free(m_ptr.release());
        }
        m_size = size;
        m_ptr.reset(cuda_make_array<T>(nullptr, size));
    }

    void copy(T* h_src)
    {
        cuda_push_array(m_ptr.get(), h_src, m_size);
    }

    void fill(T x)
    {
        fill_ongpu(m_size, x, m_ptr.get());
    }

    size_t size()
    {
        return m_size;
    }
#ifdef DARKNET_MALLOCATED
    static double m_allocated;
#endif

  private:
    size_t m_size;
    std::unique_ptr<T, deleter> m_ptr;
};

} // namespace darknet
#endif
