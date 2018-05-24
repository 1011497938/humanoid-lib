#include "darknetcxx/activationlayer.hpp"
#include "darknetcxx/blas.hpp"
#include <iostream>
#ifdef DARKNET_GPU
#include "darknetcxx/cuda_utils.hpp"
#endif
#include "darknetcxx/utils.hpp"
// TODO: 3.7 finish as much layers as possible
using std::move;
namespace darknet {

// todo: use initialize list
ActivationLayer::ActivationLayer(int batch_size, int input_size, ActivationType atype)
  : m_atype(atype)
  , m_batch_size(batch_size)
  , m_input_size(input_size)
  , m_output_size(input_size)
  , m_total_size(batch_size * input_size)
  , m_output(m_total_size)
  , m_delta(m_total_size)
#ifdef DARKNET_GPU
  , m_output_gpu(m_total_size)
  , m_delta_gpu(m_total_size)
#endif

{
    // Initialize deltas to 1.f [important .. ]
    m_delta.fill(1.f);
#ifdef DARKNET_GPU
    m_delta_gpu.fill(1.f);
#endif
    //  std::cout << "Activation Layer: " << m_input_size << std::endl;
    printf("ActivationLayer: %d x %d\n", m_batch_size, m_input_size);
}

/*******************************************************************************
 * cpu impl
 ******************************************************************************/

// TODO[optimize], no need to copy in inference mode

// state.input --> layer.output --> activate inpalce
void
ActivationLayer::forward(State& state)
{
    copy_cpu(m_total_size, state.input, output().get());
    Activation::activate<cpu>(m_atype, output().get(), m_total_size);
}

// layer.output --> layer.delta --> state.delta
void
ActivationLayer::backward(State& state)
{
    Activation::gradient<cpu>(m_atype, output().get(), m_total_size, delta());
    // Activation::gradient<cpu>(m_atype, output().get(), m_total_size,
    // state.delta);
    copy_cpu(m_total_size, m_delta.get(), state.delta);
}

/*******************************************************************************
 * gpu impl
 ******************************************************************************/
#ifdef DARKNET_GPU
void
ActivationLayer::forward_gpu(State& state)
{
    copy_ongpu(m_total_size, state.input, output_gpu().get());
    Activation::activate<gpu>(m_atype, output_gpu().get(), m_total_size);
}

void
ActivationLayer::backward_gpu(State& state)
{
    Activation::gradient<gpu>(m_atype, output_gpu().get(), m_total_size, delta_gpu());
    copy_ongpu(m_total_size, delta_gpu(), state.delta);
}
#endif
/*******************************************************************************
 * getters
 ******************************************************************************/

Ptr<float>&
ActivationLayer::output()
{
    return m_output;
}

float*
ActivationLayer::delta()
{
    return m_delta.get();
}

#ifdef DARKNET_GPU
Ptr_gpu<float>&
ActivationLayer::output_gpu()
{
    return m_output_gpu;
}

float*
ActivationLayer::delta_gpu()
{
    return m_delta_gpu.get();
}
#endif

std::map<std::string, int>
ActivationLayer::get_params()
{
    std::map<std::string, int> params = { { "out_h", 0 }, { "out_w", 0 }, { "out_c", 0 } };
    return params;
}

LayerType
ActivationLayer::type()
{
    return LayerType::BLANK;
}

/**********
 * Setters *
 **********/

void
ActivationLayer::set_batch(const int batch)
{
    m_batch_size = batch;
}
} // namespace darknet
