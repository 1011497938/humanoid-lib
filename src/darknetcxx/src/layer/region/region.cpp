#include "darknetcxx/region.hpp"

#include "darknetcxx/activations.hpp"
#include <cstdlib>

namespace darknet {
Region::Region(int batch, int w, int h, int n, int classes, int coords)
  : //
  m_batch(batch)
  , m_w(w)
  , m_h(h)
  , m_n(n)
  ,
  // m_classes(classes),
  // m_coords(coords),
  m_output_size(h * w * n * (classes + coords + 1))
  ,
  // m_input_size(m_output_size),
  // m_truths(30 * 5),
  m_cost(1)
  , m_biases(n * 2)
  , m_bias_updates(n * 2)
  , m_delta(batch * m_output_size)
  , m_output(batch * m_output_size)
#ifdef DARKNET_GPU
  , m_delta_gpu(batch * m_output_size)
  , m_output_gpu(batch * m_output_size)
#endif
{
    m_biases.fill(.5);
    printf("detection\n");
    srand(0);
}

Region::~Region()
{
}

void
Region::forward(State& state)
{
}
void
Region::backward(State& state)
{
}
#ifdef DARKNET_GPU
void
Region::forward_gpu(State& state)
{
}
void
Region::backward_gpu(State& state)
{
}
#endif
/*******************************************************************************
 * Getters
 ******************************************************************************/
Ptr<float>&
Region::output()
{
    return m_output;
}
Ptr<float>&
Region::delta()
{
    return m_delta;
}
#ifdef DARKNET_GPU
Ptr_gpu<float>&
Region::output_gpu()
{
    return m_output_gpu;
}
Ptr_gpu<float>&
Region::delta_gpu()
{
    return m_delta_gpu;
}
#endif
// out_params Region::get_params() {
//   return out_params(m_h, m_w, m_n, m_h * m_w * m_n);
// }

std::map<std::string, int>
Region::get_params()
{
    std::map<std::string, int> params = { { "out_h", m_h }, { "out_w", m_w }, { "out_n", m_n } };
    return params;
}

LayerType
Region::type()
{
    return LayerType::REGION;
}

/**********
 * Setters *
 **********/

void
Region::set_batch(const int batch)
{
    m_batch = batch;
}

} // namespace darknet
