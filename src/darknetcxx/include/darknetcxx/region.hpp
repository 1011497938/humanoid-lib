#pragma once

#include "darknetcxx/layer.hpp"
#include <map>
#include <string>
#ifdef DARKNET_GPU
#include "darknetcxx/cuda_utils.hpp"
#endif
#include "darknetcxx/utils.hpp"

namespace darknet {

class Region : public Layer
{
  public:
    Region(int batch, int w, int h, int n, int classes, int coords);
    ~Region();

    void forward(State& state) override; // NOLINT
    void backward(State& state) override;

#ifdef DARKNET_GPU
    void forward_gpu(State& state) override;
    void backward_gpu(State& state) override;
#endif

    /*****************************************************************************
     * Getters
     ****************************************************************************/
    Ptr<float>& output() override;
    Ptr<float>& delta();

#ifdef DARKNET_GPU
    Ptr_gpu<float>& output_gpu() override;
    Ptr_gpu<float>& delta_gpu();
#endif

    std::map<std::string, int> get_params() override;
    LayerType type() override;

    /**********
     * Setters *
     **********/

    void set_batch(const int batch) override;

  private:
    int m_batch;
    int m_w;
    int m_h;
    int m_n;
    // int m_classes;
    // int m_coords;
    int m_output_size;
    // int m_input_size;

    // int m_truths;

    /*****************************************************************************
     * Resources
     ****************************************************************************/
    Ptr<float> m_cost;
    Ptr<float> m_biases;
    Ptr<float> m_bias_updates;
    Ptr<float> m_delta;
    Ptr<float> m_output;

#ifdef DARKNET_GPU
    Ptr_gpu<float> m_delta_gpu;
    Ptr_gpu<float> m_output_gpu;
#endif
};
} // namespace darknet
