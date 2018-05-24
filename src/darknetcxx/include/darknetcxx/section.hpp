/**
* @Author: Yusu Pan <yuthon>
* @Date:   2017-03-13T11:41:29+08:00
* @Email:  xxdsox@gmail.com
* @Project: Dancer2017
* @Filename: section.hpp
* @Last modified by:   yuthon
* @Last modified time: 2017-03-13T11:41:31+08:00
* @Copyright: ZJUDancer
*/

#pragma once
#include <algorithm>
#include <map>
#include <string>
#include <vector>

namespace darknet {

/**
 * Section
 */

struct section {
  explicit section(const std::string &t) : type(t) {
  }
  std::string type;
  std::map<std::string, std::string> options;
};

/*********
 * Utils *
 *********/

/**
 * split string by separater
 * @param s src string
 * @param c separater
 * @param v dst string
 */
void split(const std::string &s, char c, std::vector<std::string> *v);

/**
 * decide whether it's a valid network cfg
 * @param  type type of sections' first child
 * @return      whether it's network
 */
bool is_network(const std::string &type);

/******************************
 * read option value with key *
 ******************************/

/**
 * get map value with key and default value
 * @param  options option map
 * @param  key     key
 * @param  def     default value
 * @return         value corresponding to key
 */
std::string get_value(const std::map<std::string, std::string> &options,
                      const std::string &key, const std::string &def);
int read_option_int(const std::map<std::string, std::string> &options,
                    const std::string &key, const int &def);
float read_option_float(const std::map<std::string, std::string> &options,
                        const std::string &key, const float &def);
std::string read_option_str(const std::map<std::string, std::string> &options,
                            const std::string &key, const std::string &def);
std::vector<int> read_option_int_list(
    const std::map<std::string, std::string> &options, const std::string &key,
    const std::vector<int> &def);
std::vector<float> read_option_float_list(
    const std::map<std::string, std::string> &options, const std::string &key,
    const std::vector<float> &def);
}  // namespace darknet
