#pragma once
#include "darknetcxx/activations.hpp"
#include "darknetcxx/layer.hpp"
#include <map>
#include <string>
#ifdef DARKNET_GPU
#include "darknetcxx/cuda_utils.hpp"
#endif
#include "darknetcxx/utils.hpp"

namespace darknet {
class Connected : public Layer
{
  public:
    Connected(int batch, int input_size, int output_size, ActivationType atype);
    ~Connected();

    void forward(State& state) override;
    void backward(State& state) override;
    void update(int batch, float learning_rate, float momentum, float decay) override;

#ifdef DARKNET_GPU
    void forward_gpu(State& state) override;
    void backward_gpu(State& state) override;

    // TODO is batch the same as m_batch
    void update_gpu(int batch, float learning_rate, float momentum, float decay) override;
#endif

    /***********
     * Getters *
     ***********/

    std::map<std::string, int> get_params() override;
    LayerType type() override;

    Ptr<float>& output() override;
    Ptr<float>& delta();
    Ptr<float>& biases();
    Ptr<float>& weights();

#ifdef DARKNET_GPU
    Ptr_gpu<float>& output_gpu() override;
    Ptr_gpu<float>& delta_gpu();
    Ptr_gpu<float>& biases_gpu();
    Ptr_gpu<float>& weights_gpu();
#endif

    void set_batch(const int batch) override;
    void push_connected_layer();

  private:
    int m_batch;
    int m_input_size;
    int m_output_size;
    ActivationType m_atype;

    Ptr<float> m_output;
    Ptr<float> m_delta;
    Ptr<float> m_weights;
    Ptr<float> m_weight_updates;
    Ptr<float> m_biases;
    Ptr<float> m_bias_updates;

#ifdef DARKNET_GPU
    Ptr_gpu<float> m_output_gpu;
    Ptr_gpu<float> m_delta_gpu;
    Ptr_gpu<float> m_weights_gpu;
    Ptr_gpu<float> m_weight_updates_gpu;
    Ptr_gpu<float> m_biases_gpu;
    Ptr_gpu<float> m_bias_updates_gpu;
#endif
};
} // namespace darknet
