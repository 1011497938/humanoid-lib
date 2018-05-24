/**
 * @Author: Yusu Pan <yuthon>
 * @Date:   2017-03-11T21:10:24+08:00
 * @Email:  xxdsox@gmail.com
 * @Project: Dancer2017
 * @Filename: image.hpp
 * @Last modified by:   yuthon
 * @Last modified time: 2017-03-16T20:52:27+08:00
 * @Copyright: ZJUDancer
 */

#pragma once
// FIXME(MWX): error in clion
#ifndef DARKNET_OPENCV
#define DARKNET_OPENCV
#endif

#ifdef DARKNET_OPENCV
#include <opencv2/opencv.hpp>
#endif
#include "darknetcxx/box.hpp"
#include "darknetcxx/utils.hpp"
#include <string>
#include <vector>

namespace darknet {
class Image
{
  public:
    explicit Image(const std::string& filename, const int& channels = 3, const bool verbose = true);
    explicit Image(const uint8_t* frame, const int& width, const int& height, const int& channels = 3, const bool verbose = false);
    Image(const int& width, const int& height, const int& channels = 3, const bool& verbose = true);
#ifdef DARKNET_OPENCV
    explicit Image(const cv::Mat& frame, const int& channels = 3, const bool verbose = true);
#endif
    ~Image();

    /**
     * resize image using stb_image_resize
     * @param out_h height of resized image
     * @param out_w width of resized image
     * @param out_c channels of resized image
     */
    void resize_stb(const int& out_h, const int& out_w, const int& out_c = 3);

    /**
     * resize image using darknet method
     * @param out_h height of resized image
     * @param out_w width of resized image
     * @param out_c channels of resized image
     */
    void resize_neo(const int& out_h, const int& out_w, const int& out_c = 3);

    /**
     * save image
     * @param filename filename of saved image
     */
    void save(const std::string& filename);

    /**
     * show image using OpenCV
     * @param window_name [description]
     */
    void show(const std::string& window_name = "predicted");

    /**
     * draw bbox on image
     * @param num        num of predicted boxes
     * @param thresh     lower thresh for probs
     * @param boxes      preficted boxes
     * @param probs      probs for boxes
     * @param label_list list of label names
     * @param classes    num of classes
     */
    void draw_detections(const int& num, const float& thresh, box* boxes, float** probs, const std::vector<std::string>& label_list, const int& classes);

    /**
     * find bbox using boxes and probs
     */
    void find_bbox(const int& num, const float& thresh, box* boxes, float** probs, const int& classes, std::vector<bbox>& ball_position);
    void find_relative_bbox(const int& num, const float& thresh, box* boxes, float** probs, const int& classes, std::vector<RelativeBBox>& ball_position);
    /**
     * get m_data
     */
    Ptr<float>& data();



#ifdef DARKNET_OPENCV
    /**
     * get Mat in OpenCV
     * @return converted Mat format
     */
    cv::Mat to_mat();

    /**
     * get data from Mat
     * @param frame Mat data
     */
    void from_mat(const cv::Mat& frame);
#endif

    void from_array(const uint8_t* frame);

  private:
    int m_h, m_w, m_c;
    Ptr<float> m_data;
    Ptr<float> temp_data;
    Ptr<float> part_data;
    // Ptr<unsigned char> mat_data;
    float get_pixel(float* data, const int& x, const int& y, const int& c, const int& im_h, const int& im_w, const int& im_c);
    void set_pixel(float* data, const int& x, const int& y, const int& c, const float& val, const int& im_h, const int& im_w, const int& im_c);
    void add_pixel(float* data, const int& x, const int& y, const int& c, const float& val, const int& im_h, const int& im_w, const int& im_c);
    void draw_box_width(int x1, int y1, int x2, int y2, const int& width, const float& r, const float& g, const float& b);
    void draw_box(int x1, int y1, int x2, int y2, const float& r, const float& g, const float& b);
    float get_color(const int& c, const int& x, const int& max);
    // void rgbgr_image(float *data);
    // void constrain_image(float *data);
};
} // namespace darknet
