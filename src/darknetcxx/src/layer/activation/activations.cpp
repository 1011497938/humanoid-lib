/**
 * @Author: Wenxing Mei <mwx36mwx>
 * @Date:   2017-03-11T21:10:24+08:00
 * @Email:  mwx36mwx@gmail.com
 * @Project: Dancer2017
 * @Filename: activations.cpp
 * @Last modified by:   yuthon
 * @Last modified time: 2017-03-14T15:42:45+08:00
 * @Copyright: ZJUDancer
 */

#include "darknetcxx/activations.hpp"
#include <string>

namespace darknet {
ActivationType
get_activation(const std::string& s)
{
    if (s == "logistic")
        return ActivationType::LOGISTIC;
    if (s == "loggy")
        return ActivationType::LOGGY;
    if (s == "elu")
        return ActivationType::ELU;
    if (s == "relu")
        return ActivationType::RELU;
    if (s == "relie")
        return ActivationType::RELIE;
    if (s == "plse")
        return ActivationType::PLSE;
    if (s == "hardtan")
        return ActivationType::HARDTAN;
    if (s == "lhtan")
        return ActivationType::LHTAN;
    if (s == "linear")
        return ActivationType::LINEAR;
    if (s == "ramp")
        return ActivationType::RAMP;
    if (s == "leaky")
        return ActivationType::LEAKY;
    if (s == "tanh")
        return ActivationType::TANH;
    if (s == "stair")
        return ActivationType::STAIR;
    // LOG(ERROR, "Couldn't find activation function %s, going with ReLU", s)
    return ActivationType::RELU;
}

template<>
void
Activation::activate<darknet::cpu>(ActivationType atype, float* array, std::size_t size)
{
    auto activate = [atype]() {
        switch (atype) {
            case LINEAR:
                return linear;
            case LOGISTIC:
                return logistic;
            case LOGGY:
                return loggy;
            case RELU:
                return relu;
            case ELU:
                return elu;
            case RELIE:
                return relie;
            case RAMP:
                return ramp;
            case LEAKY:
                return leaky;
            case TANH:
                return tanh;
            case PLSE:
                return plse;
            case STAIR:
                return stair;
            case HARDTAN:
                return hardtan;
            case LHTAN:
                return lhtan;
            default:
                // TODO(mwx36mwx): FIXME, log error here
                return relu;
        }
    }();

    for (std::size_t i = 0; i < size; ++i) {
        auto x = array[i];
        array[i] = activate(x);
    }
}

template<>
void
Activation::gradient<darknet::cpu>(ActivationType atype, const float* array, std::size_t size, float* delta)
{
    auto grad = [atype]() {
        switch (atype) {
            case LINEAR:
                return linear_grad;
            case LOGISTIC:
                return logistic_grad;
            case LOGGY:
                return loggy_grad;
            case RELU:
                return relu_grad;
            case ELU:
                return elu_grad;
            case RELIE:
                return relie_grad;
            case RAMP:
                return ramp_grad;
            case LEAKY:
                return leaky_grad;
            case TANH:
                return tanh_grad;
            case PLSE:
                return plse_grad;
            case STAIR:
                return stair_grad;
            case HARDTAN:
                return hardtan_grad;
            case LHTAN:
                return lhtan_grad;
            default:
                // TODO(mwx36mwx): FIXME, log error here
                return relu_grad;
        }
    }();

    for (std::size_t i = 0; i < size; ++i) {
        delta[i] *= grad(array[i]);
    }
}

} // namespace darknet
