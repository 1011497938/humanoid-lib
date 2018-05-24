/**
 * @Author: Wenxing Mei <mwx36mwx>
 * @Date:   2017-03-11T21:10:24+08:00
 * @Email:  mwx36mwx@gmail.com
 * @Project: Dancer2017
 * @Filename: network.cpp
 * @Last modified by:   yuthon
 * @Last modified time: 2017-03-18T22:20:56+08:00
 * @Copyright: ZJUDancer
 */

#include "darknetcxx/network.hpp"
#include "darknetcxx/convolutional.hpp"
#include <cstdio>
#include <string>
#include <sys/time.h>
#include <vector>

namespace darknet {

/*******************************************************************************
 * Policy *
 ******************************************************************************/

learning_rate_policy
get_policy(const std::string& s)
{
    if (s == "random")
        return learning_rate_policy::RANDOM;
    if (s == "poly")
        return learning_rate_policy::POLY;
    if (s == "constant")
        return learning_rate_policy::CONSTANT;
    if (s == "step")
        return learning_rate_policy::STEP;
    if (s == "exp")
        return learning_rate_policy::EXP;
    if (s == "sigmoid")
        return learning_rate_policy::SIG;
    if (s == "steps")
        return learning_rate_policy::STEPS;
    // LOG(ERROR, "Couldn't find policy %s, going with constant\n", s)
    return learning_rate_policy::CONSTANT;
}

/*******************************************************************************
 * Network *
 ******************************************************************************/

Network::Network(const std::vector<section>& sections)
{
    const section& sec = sections.front();
    if (!is_network(sec.type)) {
        // LOG(ERROR, "First section in config must be [net] or [network]")
    }
    m_seen = 0;
    m_batch = read_option_int(sec.options, "batch", 1);
    m_learning_rate = read_option_float(sec.options, "learning_rate", .001);
    m_momentum = read_option_float(sec.options, "momentum", .9);
    m_decay = read_option_float(sec.options, "decay", .0001);
    m_subdivisions = read_option_int(sec.options, "subdivisions", 1);
    m_time_steps = read_option_int(sec.options, "time_steps", 1);
    m_batch /= m_subdivisions;
    m_batch *= m_time_steps;
    // fix m_batch=0 if m_batch < m_subdivisions
    m_batch = m_batch == 0 ? 1 : m_batch;

    m_adam = read_option_int(sec.options, "adam", 0);
    if (m_adam) {
        m_B1 = read_option_float(sec.options, "B1", .9);
        m_B2 = read_option_float(sec.options, "B2", .999);
        m_eps = read_option_float(sec.options, "eps", .000001);
    }

    m_h = read_option_int(sec.options, "height", 0);
    m_w = read_option_int(sec.options, "width", 0);
    m_c = read_option_int(sec.options, "channels", 0);
    m_inputs = read_option_int(sec.options, "inputs", m_h * m_w * m_c);
    if (!m_inputs && !(m_h && m_w && m_c)) {
        // LOG(ERROR, "No input parameters supplied")
    }

    m_max_crop = read_option_int(sec.options, "max_crop", m_w * 2);
    m_min_crop = read_option_int(sec.options, "min_crop", m_w);

    m_angle = read_option_int(sec.options, "angle", 0);
    m_aspect = read_option_int(sec.options, "aspect", 1);
    m_saturation = read_option_float(sec.options, "saturation", static_cast<float>(1));
    m_exposure = read_option_float(sec.options, "exposure", static_cast<float>(1));
    m_hue = read_option_float(sec.options, "hue", static_cast<float>(0));

    m_policy = get_policy(read_option_str(sec.options, "policy", "constant"));
    m_burn_in = read_option_int(sec.options, "burn_in", 0);

    if (m_policy == learning_rate_policy::STEP) {
        m_step = read_option_int(sec.options, "step", 1);
        m_scale = read_option_float(sec.options, "scale", static_cast<float>(1));
    } else if (m_policy == learning_rate_policy::STEPS) {
        std::vector<int> steps_list({ 1 });
        std::vector<float> scales_list({ 1 });
        m_steps = read_option_int_list(sec.options, "steps", steps_list);
        m_scales = read_option_float_list(sec.options, "sclaes", scales_list);
        m_num_steps = m_steps.size();
    } else if (m_policy == learning_rate_policy::EXP) {
        m_gamma = read_option_float(sec.options, "gamma", static_cast<float>(1));
    } else if (m_policy == learning_rate_policy::SIG) {
        m_gamma = read_option_float(sec.options, "gamma", static_cast<float>(1));
        m_step = read_option_int(sec.options, "step", 1);
    } else if (m_policy == learning_rate_policy::POLY || m_policy == learning_rate_policy::RANDOM) {
        m_power = read_option_float(sec.options, "power", static_cast<float>(1));
    }

    m_max_batches = read_option_int(sec.options, "max_batches", 0);

    // TODO(corenel) where to init gpu_index?
    m_gpu_index = 0;
    m_max_workspace_size = 0;
}

Network::~Network()
{
}

/*********
 * Train *
 *********/

void
Network::forward_network()
{
    // m_state.workspace = workspace;
    Layer* l;
    for (int i = 0; i < num_layers(); i++) {
        l = m_layers[i];
        // printf("%s\n", get_layer_name(l->type()).c_str());
        m_state.index = i;
        // if (l->type() == LayerType::CONVOLUTIONAL) {
        //     m_state.workspace.resize(dynamic_cast<Convolutional*>(l)->workspace_size());
        // }
        l->forward(m_state);
        m_state.input = l->output().get();
    }
    // Ptr<float>& result = l->output();
    // for (size_t i = 0; i < result.size(); ++i) {
    //   printf("%f\n", result[i]);
    // }
}

Ptr<float>&
Network::network_predict(Ptr<float>& input)
{
    // State statecpu;
    m_state.index = 0;
    m_state.train = false;
    m_state.input = input.get();

    forward_network();
    return get_network_output();
}

#ifdef DARKNET_GPU
void
Network::forward_network_gpu()
{
    // m_state.workspace = workspace;
    // struct timeval t0, t1;
    // long elapsed;
    // gettimeofday(&t0, 0);
    // m_state.workspace_gpu.resize(m_max_workspace_size);
    // gettimeofday(&t1, 0);
    // elapsed = (t1.tv_sec - t0.tv_sec) * 1000000 + t1.tv_usec - t0.tv_usec;
    // std::cout << "resize workspace time: " << elapsed / 1000.0 << " ms" << std::endl;
    // std::cout << "workspace size: " << m_max_workspace_size << std::endl;

    Layer* l;

    for (int i = 0; i < num_layers(); i++) {
        l = m_layers[i];
        m_state.index = i;

        // if (l->type() == LayerType::CONVOLUTIONAL) {
        //     m_state->workspace_gpu.resize(dynamic_cast<Convolutional*>(l)->workspace_size_gpu());
        // }
        // gettimeofday(&t0, 0);
        l->forward_gpu(m_state);
        // gettimeofday(&t1, 0);
        // elapsed = (t1.tv_sec - t0.tv_sec) * 1000000 + t1.tv_usec - t0.tv_usec;
        // std::cout << "\t[" << i << "]" << get_layer_name(l->type()) << "forward gpu time: " << elapsed / 1000.0 << " ms" << std::endl;

        // gettimeofday(&t0, 0);
        m_state.input = l->output_gpu().get();
        // gettimeofday(&t1, 0);
        // elapsed = (t1.tv_sec - t0.tv_sec) * 1000000 + t1.tv_usec - t0.tv_usec;
        // std::cout << "\t[" << i << "]" << get_layer_name(l->type()) << "get output time: " << elapsed / 1000.0 << " ms" << std::endl;

        // log_layer_output(l->output().get(), l->output().size(), i);
        // TODO(corenel) add m_state.delta and m_state.delta_gpu
        // printf("%lu\n", l->output_gpu().size());
    }
    // printf("%lu\n", l->output_gpu().size());
    // Ptr<float> result(l->output_gpu().size());
    // cuda_pull_array(l->output_gpu().get(),
    //                 result.get(), l->output_gpu().size());
    // for (size_t i = 0; i < result.size(); ++i) {
    //   printf("%f\n", result[i]);
    // }
}

Ptr_gpu<float>&
Network::network_predict_gpu(Ptr<float>& input)
{
    // State stategpu;
    m_state.index = 0;
    m_state.train = false;

    if (m_tmp_input.size() != input.size()) {
        m_tmp_input.resize(input.size());
    }

    m_tmp_input.copy(input.get());

    m_state.input = m_tmp_input.get();

    // struct timeval t0, t1;
    // long elapsed;
    // gettimeofday(&t0, 0);
    forward_network_gpu();
    // gettimeofday(&t1, 0);
    // elapsed = (t1.tv_sec - t0.tv_sec) * 1000000 + t1.tv_usec - t0.tv_usec;
    // std::cout << "network forward gpu time: " << elapsed / 1000.0 << " ms" << std::endl;

    return get_network_output_gpu();
}
#endif

/**********
 * Getter *
 **********/

int
Network::num_layers()
{
    return m_layers.size();
}

params
Network::get_params()
{
    return params(m_h, m_w, m_c, m_batch);
}

int&
Network::seen()
{
    return m_seen;
}

void
Network::add_layers(Layer* l)
{
    m_layers.push_back(l);
}

std::vector<Layer*>&
Network::get_layers()
{
    return m_layers;
}

int
Network::get_current_batch()
{
    return (m_seen) / (m_batch * m_subdivisions);
}

float
Network::get_current_rate()
{
    int batch_num = get_current_batch();
    float rate;
    std::default_random_engine generator;
    std::uniform_real_distribution<float> distribution(0.0, 1.0);
    switch (m_policy) {
        case learning_rate_policy::CONSTANT:
            return m_learning_rate;
        case learning_rate_policy::STEP:
            return m_learning_rate * std::pow(m_scale, batch_num / m_step);
        case learning_rate_policy::STEPS:
            rate = m_learning_rate;
            for (int i = 0; i < m_num_steps; ++i) {
                if (m_steps[i] > batch_num)
                    return rate;
                rate *= m_scales[i];
                // if (steps[i] > batch_num - 1 && scales[i] > 1)
                //   reset_momentum(net);
            }
            return rate;
        case learning_rate_policy::EXP:
            return m_learning_rate * std::pow(m_gamma, batch_num);
        case learning_rate_policy::POLY:
            if (batch_num < m_burn_in)
                return m_learning_rate * std::pow(static_cast<float>(batch_num) / m_burn_in, m_power);
            return m_learning_rate * std::pow(1 - static_cast<float>(batch_num) / m_max_batches, m_power);
        case learning_rate_policy::RANDOM:
            return m_learning_rate * std::pow(distribution(generator), m_power);
        case learning_rate_policy::SIG:
            return m_learning_rate * (1. / (1. + std::exp(m_gamma * (batch_num - m_step))));
        default:
            // LOG(ERROR, "Policy is weird!\n");
            return m_learning_rate;
    }
}

Ptr<float>&
Network::get_network_output()
{
    return m_layers.back()->output();
}

#ifdef DARKNET_GPU
Ptr_gpu<float>&
Network::get_network_output_gpu()
{
    return m_layers.back()->output_gpu();
}
#endif

/**********
 * Setter *
 **********/

void
Network::set_network_batch(const int batch)
{
    m_batch = batch;
    for (auto l : m_layers) {
        l->set_batch(batch);
    }
}

/*******************************************************************************
 * State *
 ******************************************************************************/

State::State()
  : index(0)
{
}
} // namespace darknet