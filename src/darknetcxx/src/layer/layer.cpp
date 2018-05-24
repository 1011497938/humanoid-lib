/**
 * @Author: Wenxing Mei <mwx36mwx>
 * @Date:   2017-03-11T21:10:24+08:00
 * @Email:  mwx36mwx@gmail.com
 * @Project: Dancer2017
 * @Filename: layer.cpp
 * @Last modified by:   yuthon
 * @Last modified time: 2017-03-14T15:27:05+08:00
 * @Copyright: ZJUDancer
 */

#include "darknetcxx/layer.hpp"
#include <string>

namespace darknet {
Layer::Layer() = default;
Layer::~Layer() = default;

void
Layer::update(int batch, float learning_rate, float momentum, float decay)
{
}

#ifdef DARKNET_GPU
void
Layer::update_gpu(int batch, float learning_rate, float momentum, float decay)
{
}
#endif

LayerType
get_layer_type(const std::string& s)
{
    if (s == "[shortcut]")
        return LayerType::SHORTCUT;
    if (s == "[crop]")
        return LayerType::CROP;
    if (s == "[cost]")
        return LayerType::COST;
    if (s == "[detection]")
        return LayerType::DETECTION;
    if (s == "[region]")
        return LayerType::REGION;
    if (s == "[local]")
        return LayerType::LOCAL;
    if (s == "[convolutional]" || s == "[conv]")
        return LayerType::CONVOLUTIONAL;
    if (s == "[activation]")
        return LayerType::ACTIVE;
    if (s == "[net]" || s == "[network]")
        return LayerType::NETWORK;
    if (s == "[crnn]")
        return LayerType::CRNN;
    if (s == "[gru]")
        return LayerType::GRU;
    if (s == "[rnn]")
        return LayerType::RNN;
    if (s == "[conn]" || s == "[connected]")
        return LayerType::CONNECTED;
    if (s == "[max]" || s == "[maxpool]")
        return LayerType::MAXPOOL;
    if (s == "[reorg]")
        return LayerType::REORG;
    if (s == "[avg]" || s == "[avgpool]")
        return LayerType::AVGPOOL;
    if (s == "[dropout]")
        return LayerType::DROPOUT;
    if (s == "[lrn]" || s == "[normalization]")
        return LayerType::NORMALIZATION;
    if (s == "[batchnorm]")
        return LayerType::BATCHNORM;
    if (s == "[soft]" || s == "[softmax]")
        return LayerType::SOFTMAX;
    if (s == "[route]")
        return LayerType::ROUTE;

    return LayerType::BLANK;
}

std::string
get_layer_name(const LayerType& type)
{
    if (type == LayerType::SHORTCUT)
        return "[shortcut]";
    if (type == LayerType::CROP)
        return "[crop]";
    if (type == LayerType::COST)
        return "[cost]";
    if (type == LayerType::DETECTION)
        return "[detection]";
    if (type == LayerType::REGION)
        return "[region]";
    if (type == LayerType::LOCAL)
        return "[local]";
    if (type == LayerType::CONVOLUTIONAL)
        return "[convolutional]";
    if (type == LayerType::ACTIVE)
        return "[activation]";
    if (type == LayerType::NETWORK)
        return "[net]";
    if (type == LayerType::CRNN)
        return "[crnn]";
    if (type == LayerType::GRU)
        return "[gru]";
    if (type == LayerType::RNN)
        return "[rnn]";
    if (type == LayerType::CONNECTED)
        return "[conn]";
    if (type == LayerType::MAXPOOL)
        return "[maxpool]";
    if (type == LayerType::REORG)
        return "[reorg]";
    if (type == LayerType::AVGPOOL)
        return "[avgpool]";
    if (type == LayerType::DROPOUT)
        return "[dropout]";
    if (type == LayerType::NORMALIZATION)
        return "[normalization]";
    if (type == LayerType::BATCHNORM)
        return "[batchnorm]";
    if (type == LayerType::SOFTMAX)
        return "[softmax]";
    if (type == LayerType::ROUTE)
        return "[route]";

    return "";
}
} // namespace darknet
