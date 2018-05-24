#include "darknetcxx/batchnorm.hpp"
#include "darknetcxx/blas.hpp"
#include "darknetcxx/convolutional.hpp"

namespace darknet {
void
Convolutional::forward_gpu(State& state)
{
    m_output_gpu.fill(0);
    float one = 1.f;

    cudnnConvolutionForward(cudnn_handle(),
                            &one,
                            m_srcTensorDesc,
                            state.input,
                            m_weightDesc,
                            m_weights_gpu.get(),
                            m_convDesc,
                            m_fw_algo,
                            state.workspace_gpu.get(),
                            state.workspace_gpu.size(),
                            //            m_workspace_size,
                            &one,
                            m_dstTensorDesc,
                            m_output_gpu.get());

    Batchnorm::forward_gpu(*this, state);

    add_bias_gpu(m_output_gpu.get(), m_biases_gpu.get(), m_batch, m_depth, m_out_w * m_out_h);

    Activation::activate<gpu>(m_atype, m_output_gpu.get(), m_output_gpu.size());
}

void
Convolutional::backward_gpu(State& state)
{
    Activation::gradient<gpu>(m_atype, m_output_gpu.get(), m_output_gpu.size(), m_delta_gpu.get());

    backward_bias_gpu(m_bias_updates_gpu.get(), m_delta_gpu.get(), m_batch, m_depth, m_out_h * m_out_w);

    Batchnorm::backward_gpu(*this, state);

    //  float *original_input = state.input;
    float one = 1;

    cudnnConvolutionBackwardFilter(cudnn_handle(),
                                   &one,
                                   m_srcTensorDesc,
                                   state.input,
                                   m_ddstTensorDesc,
                                   m_delta_gpu.get(),
                                   m_convDesc,
                                   m_bf_algo,
                                   state.workspace_gpu.get(),
                                   state.workspace_gpu.size(),
                                   &one,
                                   m_dweightDesc,
                                   m_weight_updates_gpu.get());

    if (state.delta_gpu) {
        cudnnConvolutionBackwardData(cudnn_handle(),
                                     &one,
                                     m_weightDesc,
                                     m_weights_gpu.get(),
                                     m_ddstTensorDesc,
                                     m_delta_gpu.get(),
                                     m_convDesc,
                                     m_bd_algo,
                                     state.workspace_gpu.get(),
                                     state.workspace_gpu.size(),
                                     &one,
                                     m_dsrcTensorDesc,
                                     state.delta_gpu);
    }
}

void
Convolutional::update_gpu(int batch, float learning_rate, float momentum, float decay)
{
    /*****************************************************************************
     * update biases
     *
     * biases += bias_updates * learning_rate / batch
     * bias_update *= momentum
     ****************************************************************************/
    axpy_ongpu(m_depth, learning_rate / batch, m_bias_updates_gpu.get(), m_biases_gpu.get());

    scal_ongpu(m_depth, momentum, m_bias_updates_gpu.get());

    /*****************************************************************************
     * update scales
     * scales += scale_updates * learning_rate / batch
     * m_scale_updates *= momentum
     ****************************************************************************/

    axpy_ongpu(m_depth, learning_rate / batch, m_scale_updates_gpu.get(), m_scales_gpu.get());

    scal_ongpu(m_depth, momentum, m_scale_updates_gpu.get());

    /*****************************************************************************
     * update weights
     * m_weight_updates += m_weights * (-decay * batch)
     * m_weights += weights_updates * (learning_rate / batch)
     * m_weight_updates *= momentum
     ****************************************************************************/

    axpy_ongpu(m_num_weights, -decay * batch, m_weights_gpu.get(), m_weight_updates_gpu.get());

    axpy_ongpu(m_num_weights, learning_rate / batch, m_weight_updates_gpu.get(), m_weights_gpu.get());

    scal_ongpu(m_num_weights, momentum, m_weight_updates_gpu.get());
}

/*******************************************************************************
 * Helpers
 ******************************************************************************/

} // namespace darknet
