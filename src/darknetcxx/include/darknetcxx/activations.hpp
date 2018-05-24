/**
 * @Author: Wenxing Mei <mwx36mwx>
 * @Date:   2017-03-11T21:10:24+08:00
 * @Email:  mwx36mwx@gmail.com
 * @Project: Dancer2017
 * @Filename: activations.hpp
 * @Last modified by:   yuthon
 * @Last modified time: 2017-03-14T15:41:41+08:00
 * @Copyright: ZJUDancer
 */

#pragma once

//#include <algorithm>
#include "darknetcxx/darknet.hpp"
#include "darknetcxx/network.hpp"
#include <cmath>
#include <cstddef>
#include <string>

namespace darknet {

enum ActivationType
{
    LOGISTIC,
    RELU,
    RELIE,
    LINEAR,
    RAMP,
    TANH,
    PLSE,
    LEAKY,
    ELU,
    LOGGY,
    STAIR,
    HARDTAN,
    LHTAN,
    ActivationType_END
};

ActivationType
get_activation(const std::string& s);

class Activation
{
  public:
    template<typename xpu>
    static void activate(ActivationType atype, float* array, std::size_t size);

    template<typename xpu>
    static void gradient(ActivationType atype, const float* array, std::size_t size, float* delta);

  private:
    /*****************************************************************************
     * cpu activate and grad
     ****************************************************************************/

    static inline float logistic(float x)
    {
        return 1.f / (1.f + std::exp(-x));
    }

    static inline float logistic_grad(float x)
    {
        return (1.f - x) * x;
    }

    static inline float relu(float x)
    {
        return x * (x > 0.f);
    }

    static inline float relu_grad(float x)
    {
        return (x > 0.f);
    }

    static inline float relie(float x)
    {
        return (x > 0.f) ? x : .01f * x;
    }

    static inline float relie_grad(float x)
    {
        return (x > 0.f) ? 1.f : .01f;
    }

    static inline float linear(float x)
    {
        return x;
    }

    static inline float linear_grad(float x)
    {
        return 1.f;
    }

    static inline float ramp(float x)
    {
        return x * (x > 0.f) + .1f * x;
    }

    static inline float ramp_grad(float x)
    {
        return (x > 0.f) + .1f;
    }

    static inline float tanh(float x)
    {
        return (2.f / (1.f + std::exp(-2.f * x)) - 1.f);
    }

    static inline float tanh_grad(float x)
    {
        return 1.f - x * x;
    }

    static inline float plse(float x)
    {
        if (x < -4.f)
            return .01f * (x + 4.f);
        if (x > 4.f)
            return .01f * (x - 4.f) + 1.f;
        return .125f * x + .5f;
    }

    static inline float plse_grad(float x)
    {
        return (x < 0.f || x > 1.f) ? .01f : .125f;
    }

    static inline float leaky(float x)
    {
        return (x > 0.f) ? x : .1f * x;
    }

    static inline float leaky_grad(float x)
    {
        return (x > 0.f) ? 1.f : .1f;
    }

    static inline float elu(float x)
    {
        return (x >= 0.f) * x + (x < 0.f) * (std::exp(x) - 1.f);
    }

    static inline float elu_grad(float x)
    {
        return (x >= 0.f) + (x < 0.f) * (x + 1.f);
    }

    static inline float loggy(float x)
    {
        return 2.f / (1.f + std::exp(-x)) - 1.f;
    }

    static inline float loggy_grad(float x)
    {
        float y = (x + 1.f) / 2.f;
        return 2.f * (1.f - y) * y;
    }

    static inline float stair(float x)
    {
        int n = floor(x);
        if (n % 2 == 0)
            return floor(x / 2.f);
        else
            return (x - n) + floor(x / 2.f);
    }

    static inline float stair_grad(float x)
    {
        if (floor(x) == x)
            return 0.f;
        return 1.f;
    }

    static inline float hardtan(float x)
    {
        if (x < -1.f)
            return -1.f;
        if (x > 1.f)
            return 1.f;
        return x;
    }

    static inline float hardtan_grad(float x)
    {
        if (x > -1.f && x < 1.f)
            return 1.f;
        return 0.f;
    }

    static inline float lhtan(float x)
    {
        if (x < 0.f)
            return .001f * x;
        if (x > 1.f)
            return .001f * (x - 1.f) + 1.f;
        return x;
    }

    static inline float lhtan_grad(float x)
    {
        if (x > 0.f && x < 1.f)
            return 1.f;
        return .001f;
    }
};

template<>
void
Activation::activate<darknet::cpu>(ActivationType atype, float* array, std::size_t size);

template<>
void
Activation::gradient<darknet::cpu>(ActivationType atype, const float* array, std::size_t size, float* delta);

template<>
void
Activation::activate<darknet::gpu>(ActivationType atype, float* d_array, std::size_t size);

template<>
void
Activation::gradient<darknet::gpu>(ActivationType atype, const float* d_array, std::size_t size, float* d_delta);

} // namespace darknet
