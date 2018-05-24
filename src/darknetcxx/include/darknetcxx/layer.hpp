/**
 * @Author: Wenxing Mei <mwx36mwx>
 * @Date:   2017-03-11T21:10:24+08:00
 * @Email:  mwx36mwx@gmail.com
 * @Project: Dancer2017
 * @Filename: layer.hpp
 * @Last modified by:   yuthon
 * @Last modified time: 2017-03-13T11:50:23+08:00
 * @Copyright: ZJUDancer
 */

#pragma once
#include "darknetcxx/network.hpp"
#include <map>
#include <memory>
#include <string>
#ifdef DARKNET_GPU
#include "darknetcxx/cuda_utils.hpp"
#endif
#include "darknetcxx/utils.hpp"

namespace darknet {

enum CostType
{
    SSE,
    MASKED,
    SMOOTH
};

// typedef std::unique_ptr<float, deleter<float>> PointerCPU;
// typedef std::unique_ptr<float, decltype(&cuda_free<float>)> PointerGPU;
class State;

enum LayerType
{
    CONVOLUTIONAL,
    DECONVOLUTIONAL,
    CONNECTED,
    MAXPOOL,
    SOFTMAX,
    DETECTION,
    DROPOUT,
    CROP,
    ROUTE,
    COST,
    NORMALIZATION,
    AVGPOOL,
    LOCAL,
    SHORTCUT,
    ACTIVE,
    RNN,
    GRU,
    CRNN,
    BATCHNORM,
    NETWORK,
    XNOR,
    REGION,
    REORG,
    BLANK
};

class Layer
{
  public:
    Layer();
    virtual ~Layer();
    Layer(const Layer&) = delete;
    Layer& operator=(const Layer&) = delete;
    Layer(Layer&&) = delete;
    Layer& operator=(Layer&&) = delete;

    // TODO(mwx36mwx) type not intuitive
    virtual void forward(State&) = 0; // NOLINT
    virtual void backward(State&) = 0;
    virtual void update(int batch, float learning_rate, float momentum, float decay);

#ifdef DARKNET_GPU
    virtual void forward_gpu(State&) = 0;
    virtual void backward_gpu(State&) = 0;
    virtual void update_gpu(int batch, float learning_rate, float momentum, float decay);
#endif

    /**********
     * Getter *
     **********/

    virtual std::map<std::string, int> get_params() = 0;
    virtual LayerType type() = 0;
    virtual Ptr<float>& output() = 0;
#ifdef DARKNET_GPU
    virtual Ptr_gpu<float>& output_gpu() = 0;
#endif
    // todo Template this?
    // virtual float* output() = 0;
    // virtual float* delta() = 0;
    //  virtual float* output() = 0;

    /**********
     * Setter *
     **********/

    virtual void set_batch(const int batch) = 0;

  private:
};

/**
 * get layer type by string
 * @param  s string for layer type
 * @return   layer in type of layer_type
 */
LayerType
get_layer_type(const std::string& s);
std::string
get_layer_name(const LayerType& type);

} // namespace darknet
