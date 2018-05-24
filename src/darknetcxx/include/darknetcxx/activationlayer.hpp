/**
 * @Author: Wenxing Mei <mwx36mwx>
 * @Date:   2017-03-11T21:10:24+08:00
 * @Email:  mwx36mwx@gmail.com
 * @Project: Dancer2017
 * @Filename: activationlayer.hpp
 * @Last modified by:   yuthon
 * @Last modified time: 2017-03-17T15:57:14+08:00
 * @Copyright: ZJUDancer
 */

#pragma once
#include "darknetcxx/activations.hpp"
#include "darknetcxx/layer.hpp"
#include <functional>
#include <map>
#include <memory>
#include <string>

namespace darknet {

class ActivationLayer : public Layer
{
  public:
    explicit ActivationLayer(int batch, int inputs, ActivationType atype);

    void forward(State&) override; // NOLINT
    void backward(State&) override;

    Ptr<float>& output() override;
    float* delta();

#ifdef DARKNET_GPU
    void forward_gpu(State&) override;
    void backward_gpu(State&) override;
    Ptr_gpu<float>& output_gpu() override;
    float* delta_gpu();
#endif

    std::map<std::string, int> get_params() override;
    LayerType type() override;

    /**********
     * Setters *
     **********/

    void set_batch(const int batch) override;

  private:
    ActivationType m_atype;
    int m_batch_size;
    int m_input_size;
    int m_output_size;
    int m_total_size;

    Ptr<float> m_output;
    Ptr<float> m_delta;

#ifdef DARKNET_GPU
    Ptr_gpu<float> m_output_gpu;
    Ptr_gpu<float> m_delta_gpu;
#endif
};

} // namespace darknet
