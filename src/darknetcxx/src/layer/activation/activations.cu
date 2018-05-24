#include <cstdio>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <curand.h>

#include "darknetcxx/activations.hpp"
#include "darknetcxx/cuda_utils.hpp"

namespace darknet {

__device__ float
lhtan_activate_kernel(float x)
{
    if (x < 0.f)
        return .001f * x;
    if (x > 1.f)
        return .001f * (x - 1.f) + 1.f;
    return x;
}
__device__ float
lhtan_gradient_kernel(float x)
{
    if (x > 0.f && x < 1.f)
        return 1.f;
    return .001f;
}

__device__ float
hardtan_activate_kernel(float x)
{
    if (x < -1.f)
        return -1.f;
    if (x > 1.f)
        return 1.f;
    return x;
}

__device__ float
linear_activate_kernel(float x)
{
    return x;
}

__device__ float
logistic_activate_kernel(float x)
{
    return 1.f / (1.f + __expf(-x));
}
__device__ float
loggy_activate_kernel(float x)
{
    return 2.f / (1.f + __expf(-x)) - 1.f;
}

__device__ float
relu_activate_kernel(float x)
{
    return x * (x > 0.f);
}

__device__ float
elu_activate_kernel(float x)
{
    return (x >= 0.f) * x + (x < 0.f) * (__expf(x) - 1.f);
}
__device__ float
relie_activate_kernel(float x)
{
    return (x > 0.f) ? x : .01f * x;
}
__device__ float
ramp_activate_kernel(float x)
{
    return x * (x > 0.f) + .1f * x;
}
__device__ float
leaky_activate_kernel(float x)
{
    return (x > 0.f) ? x : .1f * x;
}
__device__ float
tanh_activate_kernel(float x)
{
    return (2.f / (1.f + __expf(-2.f * x)) - 1.f);
}
__device__ float
plse_activate_kernel(float x)
{
    if (x < -4.f)
        return .01f * (x + 4.f);
    if (x > 4.f)
        return .01f * (x - 4.f) + 1.f;
    return .125f * x + .5f;
}
__device__ float
stair_activate_kernel(float x)
{
    int n = floor(x);
    if (n % 2 == 0)
        return floor(x / 2.f);
    else
        return (x - n) + floor(x / 2.f);
}

__device__ float
hardtan_gradient_kernel(float x)
{
    if (x > -1.f && x < 1.f)
        return 1.f;
    return 0.f;
}
__device__ float
linear_gradient_kernel(float x)
{
    return 1.f;
}
__device__ float
logistic_gradient_kernel(float x)
{
    return (1.f - x) * x;
}
__device__ float
loggy_gradient_kernel(float x)
{
    float y = (x + 1.f) / 2.f;
    return 2.f * (1.f - y) * y;
}
__device__ float
relu_gradient_kernel(float x)
{
    return (x > 0.f);
}
__device__ float
elu_gradient_kernel(float x)
{
    return (x >= 0.f) + (x < 0.f) * (x + 1.f);
}
__device__ float
relie_gradient_kernel(float x)
{
    return (x > 0.f) ? 1.f : .01f;
}
__device__ float
ramp_gradient_kernel(float x)
{
    return (x > 0.f) + .1f;
}
__device__ float
leaky_gradient_kernel(float x)
{
    return (x > 0.f) ? 1.f : .1f;
}
__device__ float
tanh_gradient_kernel(float x)
{
    return 1.f - x * x;
}
__device__ float
plse_gradient_kernel(float x)
{
    return (x < 0.f || x > 1.f) ? .01f : .125f;
}
__device__ float
stair_gradient_kernel(float x)
{
    if (floor(x) == x)
        return 0.f;
    return 1.f;
}

__device__ float
activate_kernel(float x, ActivationType a)
{
    switch (a) {
        case LINEAR:
            return linear_activate_kernel(x);
        case LOGISTIC:
            return logistic_activate_kernel(x);
        case LOGGY:
            return loggy_activate_kernel(x);
        case RELU:
            return relu_activate_kernel(x);
        case ELU:
            return elu_activate_kernel(x);
        case RELIE:
            return relie_activate_kernel(x);
        case RAMP:
            return ramp_activate_kernel(x);
        case LEAKY:
            return leaky_activate_kernel(x);
        case TANH:
            return tanh_activate_kernel(x);
        case PLSE:
            return plse_activate_kernel(x);
        case STAIR:
            return stair_activate_kernel(x);
        case HARDTAN:
            return hardtan_activate_kernel(x);
        case LHTAN:
            return lhtan_activate_kernel(x);
    }
    return 0;
}

__device__ float
gradient_kernel(float x, ActivationType a)
{
    switch (a) {
        case LINEAR:
            return linear_gradient_kernel(x);
        case LOGISTIC:
            return logistic_gradient_kernel(x);
        case LOGGY:
            return loggy_gradient_kernel(x);
        case RELU:
            return relu_gradient_kernel(x);
        case ELU:
            return elu_gradient_kernel(x);
        case RELIE:
            return relie_gradient_kernel(x);
        case RAMP:
            return ramp_gradient_kernel(x);
        case LEAKY:
            return leaky_gradient_kernel(x);
        case TANH:
            return tanh_gradient_kernel(x);
        case PLSE:
            return plse_gradient_kernel(x);
        case STAIR:
            return stair_gradient_kernel(x);
        case HARDTAN:
            return hardtan_gradient_kernel(x);
        case LHTAN:
            return lhtan_gradient_kernel(x);
    }
    return 0;
}

__global__ void
activate_array_kernel(float* x, const int n, ActivationType a)
{
    int idx = (blockIdx.x * gridDim.y + blockIdx.y) * blockDim.x + threadIdx.x;
    if (idx < n)
        x[idx] = activate_kernel(x[idx], a);
}

__global__ void
gradient_array_kernel(const float* const x, int n, ActivationType a, float* delta)
{
    int idx = (blockIdx.x * gridDim.y + blockIdx.y) * blockDim.x + threadIdx.x;
    if (idx < n)
        delta[idx] *= gradient_kernel(x[idx], a);
}

template<>
void
Activation::activate<darknet::gpu>(ActivationType atype, float* d_array, std::size_t size)
{
    activate_array_kernel<<<cuda_gridsize(size), BLOCKSIZE>>>(d_array, size, atype);
    checkCudaErrors(cudaGetLastError());
    // check_error(cudaPeekAtLastError());
}

template<>
void
Activation::gradient<darknet::gpu>(ActivationType atype, const float* d_array, std::size_t size, float* d_delta)
{
    gradient_array_kernel<<<cuda_gridsize(size), BLOCKSIZE>>>(d_array, size, atype, d_delta);
    checkCudaErrors(cudaGetLastError());
    // check_error(cudaPeekAtLastError());
}

} // namespace darknet
