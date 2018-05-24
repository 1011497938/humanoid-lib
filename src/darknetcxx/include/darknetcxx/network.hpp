/**
 * @Author: Wenxing Mei <mwx36mwx>
 * @Date:   2017-03-11T21:10:24+08:00
 * @Email:  mwx36mwx@gmail.com
 * @Project: Dancer2017
 * @Filename: network.hpp
 * @Last modified by:   yuthon
 * @Last modified time: 2017-03-18T22:21:08+08:00
 * @Copyright: ZJUDancer
 */

#pragma once
#include "darknetcxx/layer.hpp"
#include "darknetcxx/section.hpp"
#include <memory>
#include <string>
#include <vector>
#ifdef DARKNET_GPU
#include "darknetcxx/cuda_utils.hpp"
#endif
#include "darknetcxx/utils.hpp"

namespace darknet {

/*******************************************************************************
 * Policy *
 ******************************************************************************/

enum learning_rate_policy
{
    CONSTANT,
    STEP,
    EXP,
    POLY,
    STEPS,
    SIG,
    RANDOM
};

struct params
{
    explicit params(const int height, const int width, const int channels, const int batches)
      : h(height)
      , w(width)
      , c(channels)
      , batch(batches)
    {
    }
    int h;
    int w;
    int c;
    int batch;
    int inputs;
};

/**
 * get policy by string
 * @param  s string for policy
 * @return   policy in type of learning_rate_policy
 */
learning_rate_policy
get_policy(const std::string& s);

class Layer;

/*******************************************************************************
 * State *
 ******************************************************************************/

class State
{
  public: // !? RAII
    State();
    float* truth = nullptr;
    float* input = nullptr;
    float* delta = nullptr; // TODO(mwx36mwx) OWNER??
#ifdef DARKNET_GPU
    float* delta_gpu = nullptr;
#endif

    //  float* workspace = nullptr;
    Ptr<float> workspace;
#ifdef DARKNET_GPU
    Ptr_gpu<float> workspace_gpu;
#endif
    bool train = false;
    int index;
};

/*******************************************************************************
 * Network *
 ******************************************************************************/

class Network
{
  public:
    explicit Network(const std::vector<section>& sections);
    ~Network();

    /***********
     * General *
     ***********/

    Ptr<float> workspace; // workspace for matrix manipulation

#ifdef DARKNET_GPU
    Ptr_gpu<float> input_gpu;
    Ptr_gpu<float> truth_gpu;
#endif

    /*********
     * Train *
     *********/

    void forward_network();
#ifdef DARKNET_GPU
    void forward_network_gpu();
#endif
    // void backward_network(State& state);
    // void update_network();
    // void get_network_output();
    // float get_network_cost(state);
    // int get_predicted_class_network(State net_state);

    /**********
     * Getter *
     **********/

    int num_layers(); // number of layers in network
    void add_layers(Layer* l);
    std::vector<Layer*>& get_layers();
    params get_params();
    int& seen();
    int get_current_batch();
    float get_current_rate();
    Ptr<float>& get_network_output();
    Ptr<float>& network_predict(Ptr<float>& input);
#ifdef DARKNET_GPU
    Ptr_gpu<float>& get_network_output_gpu();
    Ptr_gpu<float>& network_predict_gpu(Ptr<float>& input);
#endif
    /**********
     * Setter *
     **********/

    void set_network_batch(const int batch);
    size_t m_max_workspace_size;
    State m_state;

  private:
    /**************
     * Parameters *
     **************/

    int m_batch; // batch size
    int m_seen;  // number of images seen in traing process
    // int m_epoch;         // number of epoches in training process
    int m_subdivisions;           // number of sub divisions in one batch training
    int m_momentum;               // momentum param for gradient descent
    int m_decay;                  // regularisation parameter for L2-loss
    std::vector<Layer*> m_layers; // layers in network
    // int m_outputs;                     // ???
    std::vector<float> m_output; // FIXME output of layers, maybe Ptr?

    // learning rate policy
    learning_rate_policy m_policy; // learning rate policy
    float m_learning_rate;         // learning rate
    float m_gamma;                 // EXP decay for learning rate
    float m_scale;                 // STEP scale for decay learning rate
    int m_step;                    // STEP for decay learning rate
    std::vector<float> m_scales;   // STEPS scales for decay learning rate
    std::vector<int> m_steps;      // STEPS for decay learning rate
    int m_num_steps;               // number of STEPS
    float m_power;                 // power for RANDOM learning policy
    int m_time_steps;              // TODO(corenel) WTH is this...
    int m_max_batches;             // maximum iterations
    int m_burn_in;                 // params for POLY learning rate policy

    // optimization
    bool m_adam; // whether using adam
    float m_B1;  // beat1 param for adam
    float m_B2;  // beta2 param for adam
    float m_eps; // eps param for adam

    // image info and data argumation
    int m_inputs;       // size of an image with channels
    int m_h, m_w, m_c;  // height, width and channels for images
    int m_max_crop;     // max size of crop
    int m_min_crop;     // min size of crop
    float m_angle;      // rotation angle
    float m_aspect;     // rotation aspect
    float m_exposure;   // exposure adjust
    float m_saturation; // saturation adjust
    float m_hue;        // hue adjest

    int m_gpu_index; // GPU id
#ifdef DARKNET_GPU
    Ptr_gpu<float> m_tmp_input;
#endif
};

} // namespace darknet
