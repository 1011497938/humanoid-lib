/**
 * @Author: Yusu Pan <yuthon>
 * @Date:   2017-03-18T20:13:44+08:00
 * @Email:  xxdsox@gmail.com
 * @Project: Dancer2017
 * @Filename: detector.hpp
 * @Last modified by:   yuthon
 * @Last modified time: 2017-03-18T22:21:19+08:00
 * @Copyright: ZJUDancer
 */

#pragma once
#include "darknetcxx/box.hpp"
#include "darknetcxx/data.hpp"
#include "darknetcxx/image.hpp"
#include "darknetcxx/network.hpp"
#include "darknetcxx/parser.hpp"
#include "darknetcxx/section.hpp"
#include "darknetcxx/utils.hpp"
#include <string>
#include <vector>

namespace darknet {
void
test_detector(const std::string& data_cfg, const std::string& net_cfg, const std::string& weight_file, const std::string& image_file, const float& thresh = 0.2, const float& higher_thresh = 1.0);
void
demo_detector(const std::string& data_cfg,
              const std::string& net_cfg,
              const std::string& weight_file,
              const int& cam_index = 0,
              const std::string& video_file = "",
              const float& thresh = 0.2,
              const float& higher_thresh = 1.0,
              const int& frame_skip = 0,
              const std::string& prefix = "");
void
bbox_detection(Network* net, Image* im, const std::vector<std::string>& label_list, const float& thresh, const bool verbose = true);
void
obj_detection(Network* net, Image* im, const float& thresh, std::vector<RelativeBBox>& ball_position);
} // namespace darknet
