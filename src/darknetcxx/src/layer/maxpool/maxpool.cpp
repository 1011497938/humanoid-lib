#include "darknetcxx/maxpool.hpp"
#include <cfloat>
#include <iostream>

namespace darknet {

Maxpool::Maxpool(int batch, int w, int h, int c, int ksize, int stride)
  : //
  m_batch(batch)
  , m_w(w)
  , m_h(h)
  , m_c(c)
  , m_ksize(ksize)
  , m_stride(stride)
  , m_out_w(w / stride)
  , m_out_h(h / stride)
  , m_out_c(c)
  , m_input_size(w * h * c * batch)
  , m_output_size(m_out_h * m_out_w * m_out_c * batch)
  , m_indexes(m_output_size)
  , m_output(m_output_size)
  , m_delta(m_output_size)
#ifdef DARKNET_GPU
  , m_indexes_gpu(m_output_size)
  , m_output_gpu(m_output_size)
  , m_delta_gpu(m_output_size)
#endif
{
#ifdef DARKNET_TEST
    m_delta.fill(1);
#ifdef DARKNET_GPU
    m_delta_gpu.fill(1);
#endif
#endif
    printf("max          %d x %d / %d  %4d x%4d x%4d   ->  %4d x%4d x%4d\n", m_ksize, m_ksize, m_stride, m_w, m_h, m_c, m_out_w, m_out_h, m_out_c);
}

Maxpool::~Maxpool()
{
}

void
Maxpool::forward(State& state)
{
    int h = m_out_h;
    int w = m_out_w;
    int c = m_c;

    for (int b = 0; b < m_batch; ++b) {
        for (int k = 0; k < c; ++k) {
            for (int i = 0; i < h; ++i) {
                for (int j = 0; j < w; ++j) {
                    int out_index = j + w * (i + h * (k + c * b));
                    float max = -FLT_MAX;
                    int max_i = -1;
                    for (int n = 0; n < m_ksize; ++n) {
                        for (int m = 0; m < m_ksize; ++m) {
                            int cur_h = i * m_stride + n;
                            int cur_w = j * m_stride + m;
                            int index = cur_w + m_w * (cur_h + m_h * (k + b * m_c));
                            int valid = (cur_h >= 0 && cur_h < m_h && cur_w >= 0 && cur_w < m_w);
                            float val = (valid != 0) ? state.input[index] : -FLT_MAX;
                            max_i = (val > max) ? index : max_i;
                            max = (val > max) ? val : max;
                        }
                    }
                    m_output[out_index] = max;
                    m_indexes[out_index] = max_i;
                }
            }
        }
    }
}

void
Maxpool::backward(State& state)
{
    int h = m_out_h;
    int w = m_out_w;
    int c = m_c;
    for (int i = 0; i < h * w * c * m_batch; ++i) {
        int index = m_indexes[i];
        state.delta[index] += m_delta[i];
    }
}

/*******************************************************************************
 * Getters
 ******************************************************************************/

Ptr<float>&
Maxpool::output()
{
    return m_output;
}
Ptr<float>&
Maxpool::delta()
{
    return m_delta;
}

#ifdef DARKNET_GPU
Ptr_gpu<float>&
Maxpool::output_gpu()
{
    return m_output_gpu;
}
Ptr_gpu<float>&
Maxpool::delta_gpu()
{
    return m_delta_gpu;
}
#endif

std::map<std::string, int>
Maxpool::get_params()
{
    std::map<std::string, int> params = { { "out_h", m_out_h }, { "out_w", m_out_w }, { "out_c", m_out_c } };
    return params;
}

LayerType
Maxpool::type()
{
    return LayerType::MAXPOOL;
}

/**********
 * Setters *
 **********/

void
Maxpool::set_batch(const int batch)
{
    m_batch = batch;
    m_input_size = m_w * m_h * m_c * batch;
    m_output_size = m_out_h * m_out_w * m_out_c * batch;
    m_indexes.resize(m_output_size);
    m_output.resize(m_output_size);
    m_delta.resize(m_output_size);
#ifdef DARKNET_GPU
    m_indexes_gpu.resize(m_output_size);
    m_output_gpu.resize(m_output_size);
    m_delta_gpu.resize(m_output_size);
#endif
}

} // namespace darknet
