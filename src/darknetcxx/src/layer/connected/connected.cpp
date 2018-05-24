#include "darknetcxx/connected.hpp"
#include "darknetcxx/blas.hpp"
#include "darknetcxx/gemm.hpp"

namespace darknet {
Connected::Connected(int batch, int input_size, int output_size, ActivationType atype)

  : m_batch(batch)
  , m_input_size(input_size)
  , m_output_size(output_size)
  , m_atype(atype)
  ,
  /*************************************************************************
   * CPU
   ************************************************************************/
  m_output(batch * output_size)
  , m_delta(batch * output_size)
  ,

  m_weights(input_size * output_size)
  , m_weight_updates(input_size * output_size)
  ,

  m_biases(output_size)
  , m_bias_updates(output_size)
/*************************************************************************
 * gpu
 ************************************************************************/
#ifdef DARKNET_GPU
  , m_output_gpu(batch * output_size)
  , m_delta_gpu(batch * output_size)
  ,

  m_weights_gpu(input_size * output_size)
  , m_weight_updates_gpu(input_size * output_size)
  ,

  m_biases_gpu(output_size)
  , m_bias_updates_gpu(output_size)
#endif
{
    float scale = sqrt(2. / input_size);
    random_init(m_weights.get(), m_weights.size(), scale);
#ifdef DARKNET_GPU
    m_weights_gpu.copy(m_weights.get());
#endif

    m_biases.fill(0);

#ifdef DARKNET_TEST
    random_init(m_biases.get(), m_biases.size(), scale);
    random_init(m_delta.get(), m_delta.size(), scale);
    m_bias_updates.fill(1);
    m_weight_updates.fill(1);

#ifdef DARKNET_GPU
    m_biases_gpu.copy(m_biases.get());
    m_delta_gpu.copy(m_delta.get());

    m_bias_updates_gpu.fill(1);
    m_weight_updates_gpu.fill(1);
#endif
#endif

    printf("connected                          %4d   ->  %4d\n", input_size, output_size);
}

Connected::~Connected()
{
}

void
Connected::forward(State& state)
{
    m_output.fill(0);
    int m = m_batch;
    int k = m_input_size;
    int n = m_output_size;
    float* a = state.input;
    float* b = m_weights.get();
    float* c = m_output.get();
    gemm(0, 1, m, n, k, 1, a, k, b, k, 1, c, n);

    for (int i = 0; i < m_batch; ++i) {
        axpy_cpu(m_output_size, 1, m_biases.get(), m_output.get() + i * m_output_size);
    }
    Activation::activate<cpu>(m_atype, m_output.get(), m_output_size * m_batch);
}

void
Connected::backward(State& state)
{
    Activation::gradient<cpu>(m_atype, m_output.get(), m_output.size(), m_delta.get());
    //  gradient_array(m_output, m_output_size * m_batch, m_atype, m_delta);

    for (int i = 0; i < m_batch; ++i) {
        axpy_cpu(m_output_size, 1, m_delta.get() + i * m_output_size, m_bias_updates.get());
    }

    int m = m_output_size;
    int k = m_batch;
    int n = m_input_size;
    float* a = m_delta.get();
    float* b = state.input;
    float* c = m_weight_updates.get();

    gemm(1, 0, m, n, k, 1, a, m, b, n, 1, c, n);

    m = m_batch;
    k = m_output_size;
    n = m_input_size;

    a = m_delta.get();
    b = m_weights.get();
    c = state.delta;

    if (c)
        gemm(0, 0, m, n, k, 1, a, k, b, n, 1, c, n);
}

void
Connected::update(int batch, float learning_rate, float momentum, float decay)
{
    //   axpy_cpu(m_output_size, learning_rate / batch, m_bias_updates, 1,
    //   m_biases,
    //   1);
    // scal_cpu(m_output_size, momentum, m_bias_updates, 1);

    // if (m_batch_normalize) {
    //   axpy_cpu(m_output_size, learning_rate / batch, m_scale_updates, 1,
    //   m_scales,
    //   1);
    //   scal_cpu(m_output_size, momentum, m_scale_updates, 1);
    // }

    // axpy_cpu(m_input_size * m_output_size, -decay * batch, m_weights, 1,
    // m_weight_updates,
    //          1);
    // axpy_cpu(m_input_size * m_output_size, learning_rate / batch,
    // m_weight_updates,
    // 1,
    //          m_weights, 1);
    // scal_cpu(m_input_size * m_output_size, momentum, m_weight_updates, 1);
}

#ifdef DARKNET_GPU
void
Connected::forward_gpu(State& state)
{
    m_output_gpu.fill(0);

    int m = m_batch;
    int k = m_input_size;
    int n = m_output_size;
    float* a = state.input;
    float* b = m_weights_gpu.get();
    float* c = m_output_gpu.get();

    gemm_ongpu(0, 1, m, n, k, 1, a, k, b, k, 1, c, n);

    for (int i = 0; i < m_batch; ++i) {
        axpy_ongpu(m_output_size, 1, m_biases_gpu.get(), m_output_gpu.get() + i * m_output_size);
    }
    Activation::activate<gpu>(m_atype, m_output_gpu.get(), m_output_size * m_batch);
}

void
Connected::backward_gpu(State& state)
{
    //   int i;
    // constrain_ongpu(m_output_size * m_batch, 1, m_delta_gpu, 1);
    // gradient_array_ongpu(m_output_gpu, m_output_size * m_batch, m_activation,
    //                      m_delta_gpu);
    // for (i = 0; i < m_batch; ++i) {
    //   axpy_ongpu(m_output_size, 1, m_delta_gpu + i * m_output_size, 1,
    //   m_bias_updates_gpu,
    //              1);
    // }

    // if (m_batch_normalize) {
    //   backward_batchnorm_layer_gpu(l, state);
    // }

    // int m = m_output_size;
    // int k = m_batch;
    // int n = m_input_size;
    // float *a = m_delta_gpu;
    // float *b = state.input;
    // float *c = m_weight_updates_gpu;
    // gemm_ongpu(1, 0, m, n, k, 1, a, m, b, n, 1, c, n);

    // m = m_batch;
    // k = m_output_size;
    // n = m_input_size;

    // a = m_delta_gpu;
    // b = m_weights_gpu;
    // c = state.delta;

    // if (c)
    //   gemm_ongpu(0, 0, m, n, k, 1, a, k, b, n, 1, c, n);
}
#endif
std::map<std::string, int>
Connected::get_params()
{
    std::map<std::string, int> params = { { "out_h", 1 }, { "out_w", 1 }, { "out_c", m_output_size } };
    return params;
}

LayerType
Connected::type()
{
    return LayerType::CONNECTED;
}

Ptr<float>&
Connected::output()
{
    return m_output;
}

Ptr<float>&
Connected::biases()
{
    return m_biases;
}

Ptr<float>&
Connected::delta()
{
    return m_delta;
}

Ptr<float>&
Connected::weights()
{
    return m_weights;
}

#ifdef DARKNET_GPU
Ptr_gpu<float>&
Connected::output_gpu()
{
    return m_output_gpu;
}

Ptr_gpu<float>&
Connected::delta_gpu()
{
    return m_delta_gpu;
}

Ptr_gpu<float>&
Connected::biases_gpu()
{
    return m_biases_gpu;
}

Ptr_gpu<float>&
Connected::weights_gpu()
{
    return m_weights_gpu;
}
#endif

void
Connected::set_batch(const int batch)
{
    m_batch = batch;
    m_output.resize(batch * m_output_size);
    m_delta.resize(batch * m_output_size);
#ifdef DARKNET_GPU
    m_output_gpu.resize(batch * m_output_size);
    m_delta_gpu.resize(batch * m_output_size);
#endif
}

#ifdef DARKNET_GPU
void
Connected::push_connected_layer()
{
    cuda_push_array(m_weights_gpu.get(), m_weights.get(), m_input_size * m_output_size);
    cuda_push_array(m_biases_gpu.get(), m_biases.get(), m_output_size);
    cuda_push_array(m_weight_updates_gpu.get(), m_weight_updates.get(), m_input_size * m_output_size);
    cuda_push_array(m_bias_updates_gpu.get(), m_bias_updates.get(), m_output_size);
}

// TODO is batch the same as m_batch
void
Connected::update_gpu(int batch, float learning_rate, float momentum, float decay)
{
}
#endif

} // namespace darknet
