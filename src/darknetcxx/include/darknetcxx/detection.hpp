#pragma once
#include "darknetcxx/layer.hpp"
#include <cstdlib>
#include <map>
#include <string>
#ifdef DARKNET_GPU
#include "darknetcxx/cuda_utils.hpp"
#endif
#include "darknetcxx/utils.hpp"

namespace darknet {
struct box;
class Detection : public Layer
{
  public:
    Detection(int batch,
              int inputs,
              int n,
              int side,
              int classes,
              int coords,
              bool rescore,
              bool softmax,
              bool sqrt,
              float jitter,
              float object_scale,
              float noobject_scale,
              float class_scale,
              float coord_scale);
    ~Detection();

    void forward(State& state) override;
    void backward(State& state) override;

#ifdef DARKNET_GPU
    void forward_gpu(State& state) override;
    void backward_gpu(State& state) override;
#endif

    void get_detection_boxes(int w, int h, float thresh, float** probs, box* boxes, int only_objectness);

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
    void set_batch(const int batch) override;

  private:
    int m_batch;
    int m_input_size;  // ?
    int m_output_size; // ?
    int m_n;           // num bounding box predicts per grid
    int m_side;        // num grid per row/col
    int m_w;
    int m_h;
    int m_classes;
    int m_coords; // ?

    // int m_truth_size;

    /*****************************************************************************
     * Hyper parameters
     ****************************************************************************/
    bool m_rescore; // ?
    // bool m_softmax;
    bool m_sqrt; // ?

    // float m_jitter;          // ?
    float m_object_scale;   // ?
    float m_noobject_scale; // ?
    float m_class_scale;    // ?
    float m_coord_scale;    // ?

    //  bool m_random;
    //  bool m_reorg;

    /*****************************************************************************
     * Resources
     ****************************************************************************/

    //  Ptr<float> m_cost;
    float m_cost;
    Ptr<float> m_output;
    Ptr<float> m_delta;
#ifdef DARKNET_GPU
    Ptr_gpu<float> m_output_gpu;
    Ptr_gpu<float> m_delta_gpu;
#endif
};
} // namespace darknet
