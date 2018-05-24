/**
 * @Author: Yusu Pan <yuthon>
 * @Date:   2017-03-12T10:47:35+08:00
 * @Email:  xxdsox@gmail.com
 * @Project: Dancer2017
 * @Filename: parser.hpp
 * @Last modified by:   yuthon
 * @Last modified time: 2017-03-12T10:47:38+08:00
 * @Copyright: ZJUDancer
 */

#pragma once
#include "darknetcxx/activationlayer.hpp"
#include "darknetcxx/activations.hpp"
#include "darknetcxx/connected.hpp"
#include "darknetcxx/convolutional.hpp"
#include "darknetcxx/data.hpp"
#include "darknetcxx/detection.hpp"
#include "darknetcxx/maxpool.hpp"
#include "darknetcxx/network.hpp"
#include "darknetcxx/region.hpp"
#include "darknetcxx/section.hpp"
#include <map>
#include <string>
#include <typeinfo>
#include <vector>

namespace darknet {

/********
 * Core *
 ********/

/**
 * parse cfg file into Network class and initialzie every layer
 * @param  filename path for cfg file
 * @return          Network class instance
 */
Network&
parse_network_cfg(const std::string& filename);

/*********
 * Utils *
 *********/

/**
 * read cfg file
 * @param filename path for .cfg file
 * @return         list of sections
 */
std::vector<section>
read_cfg(const std::string& filename);

/**
 * read data cfg and get label name list
 * @param filename path for .cfg file
 * @return         list of label names
 */
std::vector<std::string>
read_data_cfg(const std::string& filename);

/**
 * read line option into section
 * @param  line   option line
 * @param  option option map in section instance
 * @return        whether option line is read successfully
 */
bool
read_option(const std::string& line, std::map<std::string, std::string>* option);

/*********************
 * Parser for layers *
 *********************/
/**
 * parse different layers
 * @param sections [description]
 * @param net      [description]
 */
void
parse_layers(const std::vector<section>& sections, Network* net);

Convolutional&
parse_conv(const std::map<std::string, std::string>& options, const params& p);

Maxpool&
parse_maxpool(const std::map<std::string, std::string>& options, const params& p);

Region&
parse_region(const std::map<std::string, std::string>& options, const params& p);

Connected&
parse_connected(const std::map<std::string, std::string>& options, const params& p);

Detection&
parse_detection(const std::map<std::string, std::string>& options, const params& p);

/****************************
 * Weight loader for layers *
 ****************************/
/**
 * load pretrained weights from local file
 * @param net      Network class instance
 * @param filename path for weight file
 */
void
load_weights(Network* net, const std::string& filename);

void
load_conv_weights(Layer* l, std::FILE* fs);
void
load_conn_weights(Layer* l, std::FILE* fs);
} // namespace darknet
