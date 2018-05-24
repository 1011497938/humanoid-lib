/**
 * @Author: Wenxing Mei <mwx36mwx>
 * @Date:   2017-03-15T08:56:22+08:00
 * @Email:  mwx36mwx@gmail.com
 * @Project: Dancer2017
 * @Filename: maxpool.hpp
 * @Last modified by:   yuthon
 * @Last modified time: 2017-03-15T12:29:42+08:00
 * @Copyright: ZJUDancer
 */

#pragma once
#include "darknetcxx/layer.hpp"
#include <map>
#include <string>
#ifdef DARKNET_GPU
#include "darknetcxx/cuda_utils.hpp"
#endif
#include "darknetcxx/utils.hpp"

// TODO(mwx36mwx) FIXME original darknet maxpool forward impl is wrong!

namespace darknet {

class Maxpool : public Layer
{
  public:
    Maxpool(int batch, int w, int h, int c, int ksize, int stride);
    ~Maxpool();

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
    int m_c;
    int m_ksize;
    int m_stride;

    int m_out_w;
    int m_out_h;
    int m_out_c;

    int m_input_size;
    int m_output_size;

    /*****************************************************************************
     * Resources
     ****************************************************************************/
    Ptr<int> m_indexes;
    Ptr<float> m_output;
    Ptr<float> m_delta;

#ifdef DARKNET_GPU
    Ptr_gpu<int> m_indexes_gpu;
    Ptr_gpu<float> m_output_gpu;
    Ptr_gpu<float> m_delta_gpu;
#endif
};
} // namespace darknet
