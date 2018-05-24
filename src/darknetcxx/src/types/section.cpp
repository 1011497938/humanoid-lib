/**
 * @Author: Yusu Pan <yuthon>
 * @Date:   2017-03-13T22:01:31+08:00
 * @Email:  xxdsox@gmail.com
 * @Project: Dancer2017
 * @Filename: section.cpp
 * @Last modified by:   yuthon
 * @Last modified time: 2017-03-13T22:01:34+08:00
 * @Copyright: ZJUDancer
 */

#include "darknetcxx/section.hpp"
#include <algorithm>
#include <map>
#include <string>
#include <vector>

namespace darknet {
void
split(const std::string& s, char c, std::vector<std::string>* v)
{
    std::string::size_type i = 0;
    std::string::size_type j = s.find(c);

    while (j != std::string::npos) {
        v->push_back(s.substr(i, j - i));
        i = ++j;
        j = s.find(c, j);

        if (j == std::string::npos) {
            v->push_back(s.substr(i, s.length()));
        }
    }
}

bool
is_network(const std::string& type)
{
    return type == "[net]" || type == "[network]";
}

// TODO(corenel) using template failed
// template <typename T>
// T get_value(const std::map<std::string, std::string> &options,
//                       const std::string &key, const T &def) {
std::string
get_value(const std::map<std::string, std::string>& options, const std::string& key, const std::string& def)
{
    auto value = options.find(key);
    if (value == options.end()) {
        return def;
    }
    return value->second;
    // std::string type_name(typeid(T).name());
    // bool is_str(type_name.find("string") != std::string::npos);
    //
    // if (std::is_integral<T>::value) {
    //   return std::stoi(value->second);
    // } else if (std::is_floating_point<T>::value) {
    //   return std::stof(value->second);
    // } else if (is_str) {
    //   return value->second;
    // } else {
    //   return def;
    // }
}

int
read_option_int(const std::map<std::string, std::string>& options, const std::string& key, const int& def)
{
    return std::stoi(get_value(options, key, std::to_string(def)));
}

float
read_option_float(const std::map<std::string, std::string>& options, const std::string& key, const float& def)
{
    return std::stof(get_value(options, key, std::to_string(def)));
}

std::string
read_option_str(const std::map<std::string, std::string>& options, const std::string& key, const std::string& def)
{
    return get_value(options, key, def);
}

// template <typename T>
std::vector<int>
read_option_int_list(const std::map<std::string, std::string>& options, const std::string& key, const std::vector<int>& def)
{
    std::string result(get_value(options, key, "NONE"));

    if (result == "NONE") {
        return def;
    } else {
        std::vector<std::string> splited;
        std::vector<int> converted;

        split(result, ',', &splited);
        std::transform(splited.begin(), splited.end(), std::back_inserter(converted), [](const std::string& str) { return std::stoi(str); });
        return converted;
    }
}

// template <typename T>
std::vector<float>
read_option_float_list(const std::map<std::string, std::string>& options, const std::string& key, const std::vector<float>& def)
{
    std::string result(get_value(options, key, "NONE"));

    if (result == "NONE") {
        return def;
    } else {
        std::vector<std::string> splited;
        std::vector<float> converted;

        split(result, ',', &splited);
        std::transform(splited.begin(), splited.end(), std::back_inserter(converted), [](const std::string& str) { return std::stof(str); });
        return converted;
    }
}
} // namespace darknet
