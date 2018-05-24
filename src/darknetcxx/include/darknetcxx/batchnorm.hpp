// reference:
// http://kratzert.github.io/2016/02/12/understanding-the-gradient-flow-through-the-batch-normalization-layer.html

#pragma once

#include "darknetcxx/layer.hpp"

namespace darknet {
class Batchnorm
{
  public:
    template<typename T>
    static inline void forward(T& l, State& state)
    {
        // if(layer.type == CONNECTED) {

        // }

        if (state.train) {
            mean_cpu(l.m_output.get(), l.m_batch, l.m_out_c, l.m_out_h * l.m_out_w, l.m_mean.get());
            variance_cpu(l.m_output.get(), l.m_mean.get(), l.m_batch, l.m_out_c, l.m_out_h * l.m_out_w, l.m_variance.get());

            scal_cpu(l.m_out_c, .99, l.m_rolling_mean.get());
            axpy_cpu(l.m_out_c, .01, l.m_mean.get(), l.m_rolling_mean.get());
            scal_cpu(l.m_out_c, .99, l.m_rolling_variance.get());
            axpy_cpu(l.m_out_c, .01, l.m_variance.get(), l.m_rolling_variance.get());

            copy_cpu(l.m_output_size * l.m_batch, l.m_output.get(), l.m_x.get());
            normalize_cpu(l.m_output.get(), l.m_mean.get(), l.m_variance.get(), l.m_batch, l.m_out_c, l.m_out_h * l.m_out_w);
            copy_cpu(l.m_output_size * l.m_batch, l.m_output.get(), l.m_x_norm.get());
        } else {
            normalize_cpu(l.m_output.get(), l.m_rolling_mean.get(), l.m_rolling_variance.get(), l.m_batch, l.m_out_c, l.m_out_h * l.m_out_w);
        }
        scale_bias(l.m_output.get(), l.m_scales.get(), l.m_batch, l.m_out_c, l.m_out_h * l.m_out_w);
    }

    template<typename T>
    static inline void backward(T& l, State& state)
    {
        backward_scale_cpu(l.m_x_norm.get(), l.m_delta.get(), l.m_batch, l.m_out_c, l.m_out_w * l.m_out_h, l.m_scale_updates.get());

        scale_bias(l.m_delta.get(), l.m_scales.get(), l.m_batch, l.m_out_c, l.m_out_h * l.m_out_w);

        mean_delta_cpu(l.m_delta.get(), l.m_variance.get(), l.m_batch, l.m_out_c, l.m_out_w * l.m_out_h, l.m_mean_delta.get());

        variance_delta_cpu(l.m_x.get(), l.m_delta.get(), l.m_mean.get(), l.m_variance.get(), l.m_batch, l.m_out_c, l.m_out_w * l.m_out_h, l.m_variance_delta.get());

        normalize_delta_cpu(l.m_x.get(), l.m_mean.get(), l.m_variance.get(), l.m_mean_delta.get(), l.m_variance_delta.get(), l.m_batch, l.m_out_c, l.m_out_w * l.m_out_h, l.m_delta.get());
    }

#ifdef DARKNET_GPU
    template<typename T>
    static inline void forward_gpu(T& l, State& state)
    {
        // if (l.type == CONNECTED) {
        // }

        if (state.train) {
            fast_mean_gpu(l.m_output_gpu.get(), l.m_batch, l.m_out_c, l.m_out_h * l.m_out_w, l.m_mean_gpu.get());
            fast_variance_gpu(l.m_output_gpu.get(), l.m_mean_gpu.get(), l.m_batch, l.m_out_c, l.m_out_h * l.m_out_w, l.m_variance_gpu.get());

            scal_ongpu(l.m_out_c, .99, l.m_rolling_mean_gpu.get());
            axpy_ongpu(l.m_out_c, .01, l.m_mean_gpu.get(), l.m_rolling_mean_gpu.get());

            scal_ongpu(l.m_out_c, .99, l.m_rolling_variance_gpu.get());
            axpy_ongpu(l.m_out_c, .01, l.m_variance_gpu.get(), l.m_rolling_variance_gpu.get());

            copy_ongpu(l.m_output_size * l.m_batch, l.m_output_gpu.get(), l.m_x_gpu.get());

            normalize_gpu(l.m_output_gpu.get(), l.m_mean_gpu.get(), l.m_variance_gpu.get(), l.m_batch, l.m_out_c, l.m_out_h * l.m_out_w);

            copy_ongpu(l.m_output_size * l.m_batch, l.m_output_gpu.get(), l.m_x_norm_gpu.get());

        } else {
            normalize_gpu(l.m_output_gpu.get(), l.m_rolling_mean_gpu.get(), l.m_rolling_variance_gpu.get(), l.m_batch, l.m_out_c, l.m_out_h * l.m_out_w);
        }
        scale_bias_gpu(l.m_output_gpu.get(), l.m_scales_gpu.get(), l.m_batch, l.m_out_c, l.m_out_h * l.m_out_w);
    }

    template<typename T>
    static inline void backward_gpu(T& l, State& state)
    {
        backward_scale_gpu(l.m_x_norm_gpu.get(), l.m_delta_gpu.get(), l.m_batch, l.m_out_c, l.m_out_w * l.m_out_h, l.m_scale_updates_gpu.get());

        scale_bias_gpu(l.m_delta_gpu.get(), l.m_scales_gpu.get(), l.m_batch, l.m_out_c, l.m_out_h * l.m_out_w);

        fast_mean_delta_gpu(l.m_delta_gpu.get(), l.m_variance_gpu.get(), l.m_batch, l.m_out_c, l.m_out_w * l.m_out_h, l.m_mean_delta_gpu.get());
        fast_variance_delta_gpu(l.m_x_gpu.get(), l.m_delta_gpu.get(), l.m_mean_gpu.get(), l.m_variance_gpu.get(), l.m_batch, l.m_out_c, l.m_out_w * l.m_out_h, l.m_variance_delta_gpu.get());
        normalize_delta_gpu(
          l.m_x_gpu.get(), l.m_mean_gpu.get(), l.m_variance_gpu.get(), l.m_mean_delta_gpu.get(), l.m_variance_delta_gpu.get(), l.m_batch, l.m_out_c, l.m_out_w * l.m_out_h, l.m_delta_gpu.get());
    }
#endif
};

} // namespace darknet
