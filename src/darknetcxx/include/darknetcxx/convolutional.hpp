/**
 * @Author: Wenxing Mei <mwx36mwx>
 * @Date:   2017-03-15T08:56:22+08:00
 * @Email:  mwx36mwx@gmail.com
 * @Project: Dancer2017
 * @Filename: convolutional.hpp
 * @Last modified by:   yuthon
 * @Last modified time: 2017-03-17T15:59:36+08:00
 * @Copyright: ZJUDancer
 */

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

// Reference: http://cs231n.github.io/convolutional-networks/
class Convolutional : public Layer
{
  public:
    friend class Batchnorm;
    Convolutional(int batch, int w, int h, int c, int depth, int ksize, int stride, int pad, ActivationType atype, bool binary, bool xnor, bool adam);

    ~Convolutional();

    void forward(State& state) override;
    void backward(State& state) override;
    void update(int batch, float learning_rate, float momentum, float decay) override;

#ifdef DARKNET_GPU
    void forward_gpu(State& state) override;
    void backward_gpu(State& state) override;

    // TODO is batch the same as m_batch
    void update_gpu(int batch, float learning_rate, float momentum, float decay) override;
#endif
    /*****************************************************************************
     * Getters
     ****************************************************************************/
    Ptr<float>& output() override;
    Ptr<float>& delta();
    Ptr<float>& weights();
    Ptr<float>& biases();
    Ptr<float>& scales();
    Ptr<float>& rolling_mean();
    Ptr<float>& rolling_variance();
    Ptr<float>& weight_updates();
    size_t workspace_size();

#ifdef DARKNET_GPU
    Ptr_gpu<float>& output_gpu() override;
    Ptr_gpu<float>& delta_gpu();
    Ptr_gpu<float>& biases_gpu();
    Ptr_gpu<float>& scales_gpu();
    Ptr_gpu<float>& weights_gpu();
    Ptr_gpu<float>& weight_updates_gpu();
    size_t workspace_size_gpu();
#endif

    std::map<std::string, int> get_params() override;
    LayerType type() override;

    /**********
     * Setters *
     **********/

    void set_batch(const int batch) override;
#ifdef DARKNET_GPU
    void push_convolutional_layer();
#endif

  private:
    void denormalize();
    void resize(int w, int h);

    void backward_bias(float* bias_updates, float* delta, int batch, int n, int size);
    void push();
    void pull();
    /*****************************************************************************
     * helpers
     ****************************************************************************/
    int out_height();
    int out_width();

  private:
    /*****************************************************************************
     * Input parameters
     ****************************************************************************/
    int m_batch;
    int m_w;
    int m_h;
    int m_c;
    int m_input_size;

    /*****************************************************************************
     * Neuron parameters
     ****************************************************************************/
    int m_depth;
    int m_ksize; // receptive field
    int m_stride;
    int m_pad;
    ActivationType m_atype;

    /*****************************************************************************
     * Output size
     ****************************************************************************/
    int m_out_h;
    int m_out_w;
    int m_out_c;
    int m_output_size;

    /*****************************************************************************
     * other parameters
     ****************************************************************************/

    int m_num_weights;
    float m_scale;
    bool m_binary;
    bool m_xnor;
    bool m_adam;
    size_t m_workspace_size;
#ifdef DARKNET_GPU
    size_t m_workspace_size_gpu;
#endif
    /*****************************************************************************
     * Resources cpu
     ****************************************************************************/
    Ptr<float> m_weights;
    Ptr<float> m_weight_updates;
    Ptr<float> m_biases;
    Ptr<float> m_bias_updates;

    Ptr<float> m_output;
    Ptr<float> m_delta;

    Ptr<float> m_scales;
    Ptr<float> m_scale_updates;

    // batch normalize
    Ptr<float> m_mean;
    Ptr<float> m_mean_delta;
    Ptr<float> m_variance;
    Ptr<float> m_variance_delta;
    Ptr<float> m_rolling_mean;
    Ptr<float> m_rolling_variance;

    Ptr<float> m_x;
    Ptr<float> m_x_norm;

    // adam
    Ptr<float> m_m;
    Ptr<float> m_v;

/*****************************************************************************
 * Resources gpu
 ****************************************************************************/
#ifdef DARKNET_GPU
    Ptr_gpu<float> m_weights_gpu;
    Ptr_gpu<float> m_weight_updates_gpu;
    Ptr_gpu<float> m_biases_gpu;
    Ptr_gpu<float> m_bias_updates_gpu;
    Ptr_gpu<float> m_output_gpu;
    Ptr_gpu<float> m_delta_gpu;
    Ptr_gpu<float> m_scales_gpu;
    Ptr_gpu<float> m_scale_updates_gpu;

    // batch normalize
    Ptr_gpu<float> m_mean_gpu;
    Ptr_gpu<float> m_mean_delta_gpu;
    Ptr_gpu<float> m_variance_gpu;
    Ptr_gpu<float> m_variance_delta_gpu;
    Ptr_gpu<float> m_rolling_mean_gpu;
    Ptr_gpu<float> m_rolling_variance_gpu;

    Ptr_gpu<float> m_x_gpu;
    Ptr_gpu<float> m_x_norm_gpu;

    // adam
    Ptr_gpu<float> m_m_gpu;
    Ptr_gpu<float> m_v_gpu;

    /*****************************************************************************
     * CUDNN
     ****************************************************************************/
    cudnnTensorDescriptor_t m_srcTensorDesc;
    cudnnTensorDescriptor_t m_dstTensorDesc;
    cudnnTensorDescriptor_t m_dsrcTensorDesc;
    cudnnTensorDescriptor_t m_ddstTensorDesc;

    cudnnFilterDescriptor_t m_weightDesc;
    cudnnFilterDescriptor_t m_dweightDesc;

    cudnnConvolutionDescriptor_t m_convDesc;
    cudnnConvolutionFwdAlgo_t m_fw_algo;
    cudnnConvolutionBwdDataAlgo_t m_bd_algo;
    cudnnConvolutionBwdFilterAlgo_t m_bf_algo;
#endif
};

} // namespace darknet
