/**
 * @Author: Yusu Pan <yuthon>
 * @Date:   2017-03-18T12:59:40+08:00
 * @Email:  xxdsox@gmail.com
 * @Project: Dancer2017
 * @Filename: data.hpp
 * @Last modified by:   yuthon
 * @Last modified time: 2017-03-18T20:39:02+08:00
 * @Copyright: ZJUDancer
 */

#pragma once
#include <string>
#include <vector>

namespace darknet {
class Data {
 public:
  Data();
  ~Data();

 private:
  // void *data;
};

std::vector<std::string> get_labels(const std::string &filename);

}  // namespace darknet
