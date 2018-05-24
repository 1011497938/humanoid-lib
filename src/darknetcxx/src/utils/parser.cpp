/**
 * @Author: Yusu Pan <yuthon>
 * @Date:   2017-03-12T10:47:48+08:00
 * @Email:  xxdsox@gmail.com
 * @Project: Dancer2017
 * @Filename: parser.cpp
 * @Last modified by:   yuthon
 * @Last modified time: 2017-03-12T10:47:49+08:00
 * @Copyright: ZJUDancer
 */

#include "darknetcxx/parser.hpp"
#include <cstdio>
#include <fstream>
#include <functional>
#include <map>
#include <string>
#include <vector>

namespace darknet {

/********
 * Core *
 ********/

Network&
parse_network_cfg(const std::string& filename)
{
    // read config
    std::vector<section> sections = read_cfg(filename);
    if (sections.empty()) {
        // LOG(ERROR, "Config file has no sections");
    }

    // generate network based on config
    Network* net = new Network(sections);

    parse_layers(sections, net);

    return *net;
}

/*********
 * Utils *
 *********/

std::vector<section>
read_cfg(const std::string& filename)
{
    std::ifstream input(filename);
    std::vector<section> sections;

    for (std::string line; std::getline(input, line);) {
        // LOG(DEBUG, line);
        if (!line.empty()) {
            switch (line.at(0)) {
                case '[':
                    sections.push_back(*new section(line));
                    break;
                case '\0':
                case '#':
                case ';':
                    break;
                default:
                    if (!read_option(line, &(sections.back().options))) {
                        // LOG(ERROR, "Config file error line %s", line)
                    }
            }
            // std::cout << line << std::endl;
        }
    }

    return sections;
}

std::vector<std::string>
read_data_cfg(const std::string& filename)
{
    std::vector<section> data_opts = read_cfg(filename);
    std::string label_file = read_option_str(data_opts.back().options, "names", "data/names.list");
    return get_labels(fix_path(label_file));
}

bool
read_option(const std::string& line, std::map<std::string, std::string>* option)
{
    std::vector<std::string> splited;
    split(line, '=', &splited);
    if (splited[0].size() + splited[1].size() == line.size() - 1) {
        (*option)[splited[0]] = splited[1];
        return true;
    } else {
        return false;
    }
}

/*********************
 * Parser for layers *
 *********************/

void
parse_layers(const std::vector<section>& sections, Network* net)
{
    params p(net->get_params());
    printf("layer     filters    size              input                output\n");
    int layer_count = 0;
    for (auto sec : sections) {
        LayerType lt(get_layer_type(sec.type));
        Layer* l;
        if (layer_count != 0) {
            printf("%5d ", layer_count);
        }
        layer_count++;

        if (lt == LayerType::CONVOLUTIONAL) {
            l = &parse_conv(sec.options, p);
#ifdef DARKNET_GPU
            size_t workspace_size = dynamic_cast<Convolutional*>(l)->workspace_size_gpu();
#else
            size_t workspace_size = dynamic_cast<Convolutional*>(l)->workspace_size();
#endif
            if (workspace_size > net->m_max_workspace_size) {
                net->m_max_workspace_size = workspace_size;
            }
        } else if (lt == LayerType::MAXPOOL) {
            l = &parse_maxpool(sec.options, p);
        } else if (lt == LayerType::REGION) {
            l = &parse_region(sec.options, p);
        } else if (lt == LayerType::CONNECTED) {
            l = &parse_connected(sec.options, p);
        } else if (lt == LayerType::DETECTION) {
            l = &parse_detection(sec.options, p);
        } else {
            continue;
        }
        // printf("%s\n", get_layer_name(l->type()).c_str());
        net->add_layers(l);
        std::map<std::string, int> op(l->get_params());
        p.h = op["out_h"];
        p.w = op["out_w"];
        p.c = op["out_c"];
        p.inputs = op["out_h"] * op["out_w"] * op["out_c"];
    }
// Resize max workspace for state
#ifdef DARKNET_GPU
    net->m_state.workspace_gpu.resize(net->m_max_workspace_size);
#else
    net->m_state.workspace.resize(net->m_max_workspace_size);
#endif
}

Convolutional&
parse_conv(const std::map<std::string, std::string>& options, const params& p)
{
    int depth = read_option_int(options, "filters", 1);
    int ksize = read_option_int(options, "size", 1);
    int stride = read_option_int(options, "stride", 1);
    int pad = read_option_int(options, "pad", 0);
    int padding = read_option_int(options, "padding", 0);
    if (pad) {
        padding = ksize / 2;
    }

    ActivationType atype = get_activation(read_option_str(options, "activation", "logistic"));
    // conv layer already comes with BN layer, so no need for
    // int batch_normalize = read_option_int(options, "batch_normalize", 0);
    bool binary = read_option_int(options, "binary", 0);
    bool xnor = read_option_int(options, "xnor", 0);

    // TODO(corenel): use smart pointer
    Convolutional* conv = new Convolutional(p.batch, p.w, p.h, p.c, depth, ksize, stride, padding, atype, binary, xnor, false);
    // TODO(corenel): set params for adam && flipped && dot
    // Never used
    return *conv;
}

Maxpool&
parse_maxpool(const std::map<std::string, std::string>& options, const params& p)
{
    int stride = read_option_int(options, "stride", 1);
    int size = read_option_int(options, "size", stride);
    // int padding = read_option_int(options, "padding", (size - 1) / 2);

    Maxpool* maxp = new Maxpool(p.batch, p.w, p.h, p.c, size, stride);
    return *maxp;
}

Region&
parse_region(const std::map<std::string, std::string>& options, const params& p)
{
    int coords = read_option_int(options, "coords", 4);
    int classes = read_option_int(options, "classes", 20);
    int num = read_option_int(options, "num", 1);

    Region* region = new Region(p.batch, p.w, p.h, num, classes, coords);
    return *region;
}

Connected&
parse_connected(const std::map<std::string, std::string>& options, const params& p)
{
    int output = read_option_int(options, "output", 1);
    std::string activation_str = read_option_str(options, "activation", "logistic");
    ActivationType activation = get_activation(activation_str);
    // int batch_normalization = read_option_int(options, "batch_normalize", 0);

    Connected* connected = new Connected(p.batch, p.inputs, output, activation);
    return *connected;
}

Detection&
parse_detection(const std::map<std::string, std::string>& options, const params& p)
{
    int classes = read_option_int(options, "classes", 1);
    int coords = read_option_int(options, "coords", 1);
    int rescore = read_option_int(options, "rescore", 0);
    int side = read_option_int(options, "side", 7);
    int num = read_option_int(options, "num", 1);
    int softmax = read_option_int(options, "softmax", 0);
    int sqrt_value = read_option_int(options, "sqrt", 0);
    int jitter = read_option_float(options, "jitter", .2);

    // int max_boxes = read_option_int(options, "max", 30);
    // int random = read_option_int(options, "random", 0);
    // int reorg = read_option_int(options, "reorg", 0);
    // int forced = read_option_int(options, "forced", 0);

    int object_scale = read_option_float(options, "object_scale", 1);
    int noobject_scale = read_option_float(options, "noobject_scale", 1);
    int class_scale = read_option_float(options, "class_scale", 1);
    int coord_scale = read_option_float(options, "coord_scale", 1);

    Detection* detection = new Detection(p.batch, p.inputs, num, side, classes, coords, rescore, softmax, sqrt_value, jitter, object_scale, noobject_scale, class_scale, coord_scale);
    return *detection;
}

/****************************
 * Weight loader for layers *
 ****************************/

void
load_weights(Network* net, const std::string& filename)
{
    // #ifdef DARKNET_GPU
    //   if (net->gpu_index >= 0) {
    //     cuda_set_device(net->gpu_index);
    //   }
    // #endif
    printf("Loading weights from %s...\n", filename.c_str());
    std::FILE* fs = std::fopen(filename.c_str(), "rb");

    int major;
    int minor;
    int revision;
    std::fread(&major, sizeof(int), 1, fs);
    std::fread(&minor, sizeof(int), 1, fs);
    std::fread(&revision, sizeof(int), 1, fs);
    std::fread(&(net->seen()), sizeof(int), 1, fs);
    printf("mj = %d, mn = %d, net->seen = %d\n", major, minor, net->seen() ? net->seen() : 0);
    // int transpose = (major > 1000) || (minor > 1000);

    std::vector<Layer*>& layers(net->get_layers());
    for (auto l : layers) {
        if (l->type() == LayerType::CONVOLUTIONAL) {
            load_conv_weights(l, fs);
        } else if (l->type() == LayerType::CONNECTED) {
            load_conn_weights(l, fs);
        } else {
            continue;
        }
    }

    std::fclose(fs);
    printf("Done\n");
}

void
load_conv_weights(Layer* l, std::FILE* fs)
{
    std::map<std::string, int> params(l->get_params());

    int depth = params["depth"];
    int ksize = params["ksize"];
    int c = params["c"];

    int num_weights = depth * ksize * ksize * c;
    printf("load_convolutional_weights: depth * c * ksize * ksize = %d\n", num_weights);
    std::fread(dynamic_cast<Convolutional*>(l)->biases().get(), sizeof(float), params["depth"], fs);
    std::fread(dynamic_cast<Convolutional*>(l)->scales().get(), sizeof(float), params["depth"], fs);
    std::fread(dynamic_cast<Convolutional*>(l)->rolling_mean().get(), sizeof(float), params["depth"], fs);
    std::fread(dynamic_cast<Convolutional*>(l)->rolling_variance().get(), sizeof(float), params["depth"], fs);
    std::fread(dynamic_cast<Convolutional*>(l)->weights().get(), sizeof(float), num_weights, fs);
#ifdef DARKNET_GPU
    dynamic_cast<Convolutional*>(l)->push_convolutional_layer();
#endif
    // TODO(corenel) if conv layer has adam, read m & v from weigths.
}

void
load_conn_weights(Layer* l, std::FILE* fs)
{
    std::fread(dynamic_cast<Connected*>(l)->biases().get(), sizeof(float), dynamic_cast<Connected*>(l)->biases().size(), fs);
    std::fread(dynamic_cast<Connected*>(l)->weights().get(), sizeof(float), dynamic_cast<Connected*>(l)->weights().size(), fs);
#ifdef DARKNET_GPU
    dynamic_cast<Connected*>(l)->push_connected_layer();
#endif
}
} // namespace darknet
