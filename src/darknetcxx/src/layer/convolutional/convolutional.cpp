#include "darknetcxx/convolutional.hpp"
#include "darknetcxx/batchnorm.hpp"
#include "darknetcxx/blas.hpp"
#include "darknetcxx/gemm.hpp"
#include "darknetcxx/im2col.hpp"
#include <cmath>
#include <iostream>

namespace darknet {

Convolutional::Convolutional(int batch, int w, int h, int c, int depth,
                             int ksize, int stride, int pad,
                             ActivationType atype, bool binary, bool xnor,
                             bool adam)
    : // input
      m_batch(batch),
      m_w(w), m_h(h), m_c(c), m_input_size(h * w * c),
      // neuron
      m_depth(depth), m_ksize(ksize), m_stride(stride), m_pad(pad),
      m_atype(atype),
      // output
      m_out_h(out_height()), m_out_w(out_width()), m_out_c(depth),
      m_output_size(m_out_h * m_out_w * m_out_c),
      // misc
      m_num_weights(c * ksize * ksize * depth),
      m_scale(sqrt(2. / (ksize * ksize * c))), // wtf?
      m_binary(binary), m_xnor(xnor), m_adam(adam),
      // resources
      // cpu
      // #if !defined(DARKNET_TX1) && !defined(DARKNET_TX2)
      m_weights(m_num_weights), m_weight_updates(m_num_weights),
      m_biases(depth), m_bias_updates(depth), m_output(m_output_size * batch),
      m_delta(m_output_size * batch)
// #endif
// gpu

#ifdef DARKNET_GPU
      ,
      m_weights_gpu(m_num_weights), m_weight_updates_gpu(m_num_weights),
      m_biases_gpu(depth), m_bias_updates_gpu(depth),
      m_output_gpu(m_output_size * batch), m_delta_gpu(m_output_size * batch)
#endif

{
/*****************************************************************************
 * CPU init
 ****************************************************************************/
#ifdef DARKNET_MALLOCATED
  fprintf(stderr,
          "conv layer batch: %d out (w %d, h %d, c %d) output_size: %d\n",
          m_batch, m_out_w, m_out_h, m_out_c, m_output_size);
#endif

// TODO INIT Weights and Biases
#if !defined(DARKNET_TX1) && !defined(DARKNET_TX2)
  // #ifndef DARKNET_TEGRA
  random_init(m_weights.get(), m_weights.size(), m_scale);
#ifdef DARKNET_GPU
  m_weights_gpu.copy(m_weights.get());
#endif
#endif
  m_scales.resize(depth);
  m_scales.fill(1);
  m_scale_updates.resize(depth);

  m_mean.resize(depth);
  m_variance.resize(depth);
  m_mean_delta.resize(depth);
  m_variance_delta.resize(depth);
  m_rolling_mean.resize(depth);
  m_rolling_variance.resize(depth);
  m_x.resize(batch * m_output_size);
  m_x_norm.resize(batch * m_output_size);

/*****************************************************************************
 * GPU init
 ****************************************************************************/
#ifdef DARKNET_GPU
  m_scales_gpu.resize(depth);
  m_scales_gpu.fill(1);
  m_scale_updates_gpu.resize(depth);

  m_mean_gpu.resize(depth);
  m_variance_gpu.resize(depth);
  m_mean_delta_gpu.resize(depth);
  m_variance_delta_gpu.resize(depth);
  m_rolling_mean_gpu.resize(depth);
  m_rolling_variance_gpu.resize(depth);

  m_x_gpu.resize(batch * m_output_size);
  m_x_norm_gpu.resize(batch * m_output_size);
#endif

// init params for testing
#ifdef DARKNET_TEST
#if !defined(DARKNET_TX1) && !defined(DARKNET_TX2)
  random_init(m_biases.get(), m_biases.size(), m_scale);
  random_init(m_delta.get(), m_delta.size(), m_scale);
  random_init(m_rolling_mean.get(), m_rolling_mean.size(), m_scale);

#ifdef DARKNET_GPU
  m_biases_gpu.copy(m_biases.get());
  m_delta_gpu.copy(m_delta.get());
  m_rolling_mean_gpu.copy(m_rolling_mean.get());
#endif

  // random_init(m_rolling_variance.get(), m_rolling_variance.size(), m_scale);
  // m_rolling_variance_gpu.copy(m_rolling_variance.get());

  // m_biases.fill(0);
  // m_biases_gpu.fill(0);

  // m_delta.fill(1);
  // m_delta_gpu.fill(1);

  // m_rolling_mean.fill(1);
  // m_rolling_mean_gpu.fill(1);

  m_rolling_variance.fill(1);
  m_bias_updates.fill(1);
  m_scale_updates.fill(1);
  m_weight_updates.fill(1);

#ifdef DARKNET_GPU
  m_rolling_variance_gpu.fill(1);
  m_bias_updates_gpu.fill(1);
  m_scale_updates_gpu.fill(1);
  m_weight_updates_gpu.fill(1);
#endif
#endif
#endif

#ifdef DARKNET_GPU
  // cudnn init
  cudnnCreateTensorDescriptor(&m_srcTensorDesc);
  cudnnCreateTensorDescriptor(&m_dstTensorDesc);
  cudnnCreateTensorDescriptor(&m_dsrcTensorDesc);
  cudnnCreateTensorDescriptor(&m_ddstTensorDesc);
  cudnnCreateFilterDescriptor(&m_weightDesc);
  cudnnCreateFilterDescriptor(&m_dweightDesc);
  cudnnCreateConvolutionDescriptor(&m_convDesc);

  // setup
  cudnnSetTensor4dDescriptor(m_dsrcTensorDesc, CUDNN_TENSOR_NCHW,
                             CUDNN_DATA_FLOAT, batch, c, h, w);
  cudnnSetTensor4dDescriptor(m_ddstTensorDesc, CUDNN_TENSOR_NCHW,
                             CUDNN_DATA_FLOAT, batch, m_out_c, m_out_h,
                             m_out_w);
  cudnnSetFilter4dDescriptor(m_dweightDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
                             depth, c, ksize, ksize);

  cudnnSetTensor4dDescriptor(m_srcTensorDesc, CUDNN_TENSOR_NCHW,
                             CUDNN_DATA_FLOAT, batch, c, h, w);
  cudnnSetTensor4dDescriptor(m_dstTensorDesc, CUDNN_TENSOR_NCHW,
                             CUDNN_DATA_FLOAT, batch, m_out_c, m_out_h,
                             m_out_w);
  cudnnSetFilter4dDescriptor(m_weightDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
                             depth, c, ksize, ksize);
  cudnnSetConvolution2dDescriptor(m_convDesc, pad, pad, stride, stride, 1, 1,
                                  CUDNN_CROSS_CORRELATION);
  cudnnGetConvolutionForwardAlgorithm(
      cudnn_handle(), m_srcTensorDesc, m_weightDesc, m_convDesc,
      m_dstTensorDesc, CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, &m_fw_algo);
  cudnnGetConvolutionBackwardDataAlgorithm(
      cudnn_handle(), m_weightDesc, m_ddstTensorDesc, m_convDesc,
      m_dsrcTensorDesc, CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST, 0,
      &m_bd_algo);
  cudnnGetConvolutionBackwardFilterAlgorithm(
      cudnn_handle(), m_srcTensorDesc, m_ddstTensorDesc, m_convDesc,
      m_dweightDesc, CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST, 0,
      &m_bf_algo);

  m_workspace_size_gpu = workspace_size_gpu();
#endif
  m_workspace_size = workspace_size();

  // TODO use log
  printf("conv  %5d %2d x%2d /%2d  %4d x%4d x%4d   ->  %4d x%4d x%4d pad: %d\n",
         m_depth, m_ksize, m_ksize, m_stride, m_w, m_h, m_c, m_out_w, m_out_h,
         m_out_c, m_pad);
}

Convolutional::~Convolutional() {}

/*******************************************************************************
 * interface
 ******************************************************************************/

// convolution --> batch_norm (normalization, scale) --> add_bias --> activation
void Convolutional::forward(State &state) {
  m_output.fill(0);

  int m = m_depth;
  int k = m_ksize * m_ksize * m_c;
  int n = m_out_h * m_out_w;

  auto a = m_weights.get();
  auto b = state.workspace.get();
  auto c = m_output.get();

  for (int i = 0; i < m_batch; ++i) {
    im2col_cpu(state.input, m_c, m_h, m_w, m_ksize, m_stride, m_pad, b);
    gemm(0, 0, m, n, k, 1, a, k, b, n, 1, c, n);
    c += n * m;
    state.input += m_c * m_h * m_w;
  }

  Batchnorm::forward(*this, state);

  add_bias(m_output.get(), m_biases.get(), m_batch, m_depth, m_out_h * m_out_w);

  Activation::activate<cpu>(m_atype, m_output.get(), m_output.size());
}

// TODO batch norm

void Convolutional::backward(State &state) {
  int m = m_depth;
  int n = m_ksize * m_ksize * m_c;
  int k = m_out_h * m_out_w;

  Activation::gradient<cpu>(m_atype, m_output.get(), m_output.size(),
                            m_delta.get());

  backward_bias(m_bias_updates.get(), m_delta.get(), m_batch, m_depth, k);

  Batchnorm::backward(*this, state);

  for (int i = 0; i < m_batch; ++i) {
    float *a = m_delta.get() + i * m * k;
    float *b = state.workspace.get();
    float *c = m_weight_updates.get();

    float *im = state.input + i * m_c * m_h * m_w;

    im2col_cpu(im, m_c, m_h, m_w, m_ksize, m_stride, m_pad, b);
    gemm(0, 1, m, n, k, 1, a, k, b, k, 1, c, n);

    if (state.delta) {
      a = m_weights.get();
      b = m_delta.get() + i * m * k;
      c = state.workspace.get();
      gemm(1, 0, n, k, m, 1, a, n, b, k, 0, c, k);

      col2im_cpu(state.workspace.get(), m_c, m_h, m_w, m_ksize, m_stride, m_pad,
                 state.delta + i * m_c * m_h * m_w);
    }
  }
}

void Convolutional::update(int batch, float learning_rate, float momentum,
                           float decay) {
  /*****************************************************************************
   * update biases
   *
   * biases += bias_updates * learning_rate / batch
   * bias_update *= momentum
   ****************************************************************************/
  axpy_cpu(m_depth, learning_rate / batch, m_bias_updates.get(),
           m_biases.get());

  scal_cpu(m_depth, momentum, m_bias_updates.get());

  /*****************************************************************************
   * update scales
   * scales += scale_updates * learning_rate / batch
   * m_scale_updates *= momentum
   ****************************************************************************/

  axpy_cpu(m_depth, learning_rate / batch, m_scale_updates.get(),
           m_scales.get());

  scal_cpu(m_depth, momentum, m_scale_updates.get());

  /*****************************************************************************
   * update weights
   * m_weight_updates += m_weights * (-decay * batch)
   * m_weights += weights_updates * (learning_rate / batch)
   * m_weight_updates *= momentum
   ****************************************************************************/

  axpy_cpu(m_num_weights, -decay * batch, m_weights.get(),
           m_weight_updates.get());

  axpy_cpu(m_num_weights, learning_rate / batch, m_weight_updates.get(),
           m_weights.get());

  scal_cpu(m_num_weights, momentum, m_weight_updates.get());
}

/*******************************************************************************
 * private helpers
 ******************************************************************************/

// void Convolutional::add_bias(float* output, float* biases, int batch, int
// ksize,
//                              int size) {
//   for (int b = 0; b < batch; ++b) {
//     for (int k = 0; k < ksize; ++k) {
//       for (int j = 0; j < size; ++j) {
//         output[(b * ksize + k) * size + j] += biases[k];
//       }
//     }
//   }
// }

void Convolutional::backward_bias(float *bias_updates, float *delta, int batch,
                                  int n, int size) {
  for (int b = 0; b < batch; ++b) {
    for (int i = 0; i < n; ++i) {
      bias_updates[i] += sum_array(delta + size * (i + b * n), size);
    }
  }
}

/*******************************************************************************
 * Getters
 ******************************************************************************/

Ptr<float> &Convolutional::output() { return m_output; }
Ptr<float> &Convolutional::delta() { return m_delta; }
Ptr<float> &Convolutional::biases() { return m_biases; }
Ptr<float> &Convolutional::scales() { return m_scales; }
Ptr<float> &Convolutional::rolling_mean() { return m_rolling_mean; }
Ptr<float> &Convolutional::rolling_variance() { return m_rolling_variance; }
Ptr<float> &Convolutional::weights() { return m_weights; }
Ptr<float> &Convolutional::weight_updates() { return m_weight_updates; }

#ifdef DARKNET_GPU
Ptr_gpu<float> &Convolutional::output_gpu() { return m_output_gpu; }
Ptr_gpu<float> &Convolutional::delta_gpu() { return m_delta_gpu; }
Ptr_gpu<float> &Convolutional::biases_gpu() { return m_biases_gpu; }
Ptr_gpu<float> &Convolutional::scales_gpu() { return m_scales_gpu; }
Ptr_gpu<float> &Convolutional::weights_gpu() { return m_weights_gpu; }
Ptr_gpu<float> &Convolutional::weight_updates_gpu() {
  return m_weight_updates_gpu;
}
#endif

int Convolutional::out_height() {
  if ((m_h + 2 * m_pad - m_ksize) % m_stride != 0) {
    printf("ConvNets size error %d %d %d %d\n", m_h, m_pad, m_ksize, m_stride);
  }
  return (m_h + 2 * m_pad - m_ksize) / m_stride + 1;
}

int Convolutional::out_width() {
  if ((m_w + 2 * m_pad - m_ksize) % m_stride != 0) {
    printf("ConvNets size error %d %d %d %d\n", m_h, m_pad, m_ksize, m_stride);
  }
  return (m_w + 2 * m_pad - m_ksize) / m_stride + 1;
}

#ifdef DARKNET_GPU
size_t Convolutional::workspace_size_gpu() {
  size_t most = 0;
  size_t s = 0;
  cudnnGetConvolutionForwardWorkspaceSize(cudnn_handle(), m_srcTensorDesc,
                                          m_weightDesc, m_convDesc,
                                          m_dstTensorDesc, m_fw_algo, &s);

  if (s > most)
    most = s;

  cudnnGetConvolutionBackwardFilterWorkspaceSize(
      cudnn_handle(), m_srcTensorDesc, m_ddstTensorDesc, m_convDesc,
      m_dweightDesc, m_bf_algo, &s);

  if (s > most)
    most = s;
  cudnnGetConvolutionBackwardDataWorkspaceSize(cudnn_handle(), m_weightDesc,
                                               m_ddstTensorDesc, m_convDesc,
                                               m_dsrcTensorDesc, m_bd_algo, &s);
  if (s > most)
    most = s;
  return most; // TODO FIXME: most / sizeof(float) ??
}
#endif

size_t Convolutional::workspace_size() {
  return m_out_h * m_out_w * m_ksize * m_ksize * m_c;
}

std::map<std::string, int> Convolutional::get_params() {
  std::map<std::string, int> params = {{"out_h", m_out_h}, {"out_w", m_out_w},
                                       {"out_c", m_out_c}, {"c", m_c},
                                       {"ksize", m_ksize}, {"depth", m_depth}};
  return params;
}

LayerType Convolutional::type() { return LayerType::CONVOLUTIONAL; }

/**********
 * Setters *
 **********/

void Convolutional::set_batch(const int batch) {
  m_batch = batch;
  m_output.resize(m_output_size * batch);
  m_delta.resize(m_output_size * batch);
#ifdef DARKNET_GPU
  m_output_gpu.resize(m_output_size * batch);
  m_delta_gpu.resize(m_output_size * batch);
#endif

  m_x.resize(batch * m_output_size);
  m_x_norm.resize(batch * m_output_size);

#ifdef DARKNET_GPU
  // setup
  cudnnSetTensor4dDescriptor(m_dsrcTensorDesc, CUDNN_TENSOR_NCHW,
                             CUDNN_DATA_FLOAT, batch, m_c, m_h, m_w);
  cudnnSetTensor4dDescriptor(m_ddstTensorDesc, CUDNN_TENSOR_NCHW,
                             CUDNN_DATA_FLOAT, batch, m_out_c, m_out_h,
                             m_out_w);
  cudnnSetFilter4dDescriptor(m_dweightDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
                             m_depth, m_c, m_ksize, m_ksize);

  cudnnSetTensor4dDescriptor(m_srcTensorDesc, CUDNN_TENSOR_NCHW,
                             CUDNN_DATA_FLOAT, batch, m_c, m_h, m_w);
  cudnnSetTensor4dDescriptor(m_dstTensorDesc, CUDNN_TENSOR_NCHW,
                             CUDNN_DATA_FLOAT, batch, m_out_c, m_out_h,
                             m_out_w);
  cudnnSetFilter4dDescriptor(m_weightDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
                             m_depth, m_c, m_ksize, m_ksize);
  cudnnSetConvolution2dDescriptor(m_convDesc, m_pad, m_pad, m_stride, m_stride,
                                  1, 1, CUDNN_CROSS_CORRELATION);
  cudnnGetConvolutionForwardAlgorithm(
      cudnn_handle(), m_srcTensorDesc, m_weightDesc, m_convDesc,
      m_dstTensorDesc, CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, &m_fw_algo);
  cudnnGetConvolutionBackwardDataAlgorithm(
      cudnn_handle(), m_weightDesc, m_ddstTensorDesc, m_convDesc,
      m_dsrcTensorDesc, CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST, 0,
      &m_bd_algo);
  cudnnGetConvolutionBackwardFilterAlgorithm(
      cudnn_handle(), m_srcTensorDesc, m_ddstTensorDesc, m_convDesc,
      m_dweightDesc, CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST, 0,
      &m_bf_algo);
#endif
}

#ifdef DARKNET_GPU
void Convolutional::push_convolutional_layer() {
  cuda_push_array(m_weights_gpu.get(), m_weights.get(),
                  m_c * m_depth * m_ksize * m_ksize);
  cuda_push_array(m_biases_gpu.get(), m_biases.get(), m_depth);
  cuda_push_array(m_weight_updates_gpu.get(), m_weight_updates.get(),
                  m_c * m_depth * m_ksize * m_ksize);
  cuda_push_array(m_bias_updates_gpu.get(), m_bias_updates.get(), m_depth);
  cuda_push_array(m_scales_gpu.get(), m_scales.get(), m_depth);
  cuda_push_array(m_rolling_mean_gpu.get(), m_rolling_mean.get(), m_depth);
  cuda_push_array(m_rolling_variance_gpu.get(), m_rolling_variance.get(),
                  m_depth);
}
#endif
} // namespace darknet
