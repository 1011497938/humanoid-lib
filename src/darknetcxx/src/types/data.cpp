/**
 * @Author: Yusu Pan <yuthon>
 * @Date:   2017-03-18T12:59:40+08:00
 * @Email:  xxdsox@gmail.com
 * @Project: Dancer2017
 * @Filename: data.cpp
 * @Last modified by:   yuthon
 * @Last modified time: 2017-03-18T20:40:21+08:00
 * @Copyright: ZJUDancer
 */

#include "darknetcxx/data.hpp"
#include <fstream>
#include <string>
#include <vector>

namespace darknet {
Data::Data()
{
}

Data::~Data()
{
}

std::vector<std::string>
get_labels(const std::string& filename)
{
    std::ifstream label_file(filename);
    std::vector<std::string> label_list;

    // read label names line by line
    for (std::string line; std::getline(label_file, line);) {
        if (!line.empty()) {
            label_list.push_back(line);
        }
    }

    return label_list;
}

} // namespace darknet
