/**
 * @Author: Yusu Pan <yuthon>
 * @Date:   2017-03-11T21:10:24+08:00
 * @Email:  xxdsox@gmail.com
 * @Project: Dancer2017
 * @Filename: image.cpp
 * @Last modified by:   yuthon
 * @Last modified time: 2017-03-16T21:04:50+08:00
 * @Copyright: ZJUDancer
 */

#include <cstring>
#include <string>
#include <vector>
#define STB_IMAGE_IMPLEMENTATION
#include "darknetcxx/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "darknetcxx/stb_image_write.h"
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "darknetcxx/image.hpp"
#include "darknetcxx/stb_image_resize.h"

namespace darknet {
Image::Image(const std::string& filename, const int& channels, const bool verbose)
{
    unsigned char* image_data = stbi_load(filename.c_str(), &m_w, &m_h, &m_c, channels);
    // TODO(corenel) Log error
    if (!image_data) {
        fprintf(stderr, "Cannot load image \"%s\"\nSTB Reason: %s\n", filename.c_str(), stbi_failure_reason());
        exit(0);
    }
    if (channels)
        m_c = channels;
    m_data.resize(m_h * m_w * m_c);

    for (int k = 0; k < m_c; ++k) {
        for (int j = 0; j < m_h; ++j) {
            for (int i = 0; i < m_w; ++i) {
                int dst_index = i + m_w * j + m_w * m_h * k;
                int src_index = k + m_c * i + m_c * m_w * j;
                m_data[dst_index] = static_cast<float>(image_data[src_index]) / 255.;
            }
        }
    }

    if (verbose) {
        printf("Load image: %d x %d x %d\n", m_w, m_h, m_c);
    }

    free(image_data);
}

Image::Image(const uint8_t* frame, const int& width, const int& height, const int& channels, const bool verbose)
{
    assert(width && height && channels);
    m_w = width;
    m_h = height;
    m_c = channels;
    m_data.resize(m_h * m_w * m_c);

    for (int k = 0; k < m_c; ++k) {
        for (int j = 0; j < m_h; ++j) {
            for (int i = 0; i < m_w; ++i) {
                int dst_index = i + m_w * j + m_w * m_h * k;
                int src_index = k + m_c * i + m_c * m_w * j;
                m_data[dst_index] = static_cast<float>(frame[src_index]) / 255.;
            }
        }
    }

    if (verbose) {
        printf("Load image: %d x %d x %d\n", m_w, m_h, m_c);
    }
}

Image::Image(const int& width, const int& height, const int& channels, const bool& verbose)
  : m_h(height)
  , m_w(width)
  , m_c(channels)
  , m_data(width * height * channels)
{
    if (verbose) {
        printf("Load image: %d x %d x %d\n", m_w, m_h, m_c);
    }
}

#ifdef DARKNET_OPENCV
Image::Image(const cv::Mat& frame, const int& channels, const bool verbose)
{
    m_w = frame.cols;
    m_h = frame.rows;
    m_c = frame.channels();

    cv::Mat converted;
    cv::cvtColor(frame, converted, CV_BGR2RGB);

    if (m_data.size() != static_cast<size_t>(m_h * m_w * m_c)) {
        m_data.resize(m_h * m_w * m_c);
    }

    int count = 0;
    for (int k = 0; k < m_c; ++k) {
        for (int j = 0; j < m_h; ++j) {
            for (int i = 0; i < m_w; ++i) {
                // int dst_index = i + m_w * j + m_w * m_h * k;
                int src_index = k + m_c * i + m_c * m_w * j;
                m_data[count++] = static_cast<float>(converted.data[src_index]) / 255.;
            }
        }
    }
    // rgbgr_image(m_data.get());

    if (verbose) {
        printf("Load image: %d x %d x %d\n", m_w, m_h, m_c);
    }
}
#endif

Image::~Image()
{
}

/********************
 * image operations *
 ********************/

void
Image::resize_stb(const int& out_h, const int& out_w, const int& out_c)
{
    int size = m_w * m_h * m_c;
    // store original image
    Ptr<float> tmp_data(size);
    std::memcpy(tmp_data.get(), m_data.get(), sizeof(float) * size);
    // resize() will delete all data
    m_data.resize(out_h * out_w * out_c);
    stbir_resize_float(tmp_data.get(), m_w, m_h, 0, m_data.get(), out_w, out_h, 0, out_c);
    m_h = out_h;
    m_w = out_w;
    m_c = out_c;
}

void
Image::resize_neo(const int& out_h, const int& out_w, const int& out_c)
{
    temp_data.resize(m_w * m_h * m_c);
    std::memcpy(temp_data.get(), m_data.get(), sizeof(float) * m_w * m_h * m_c);
    m_data.resize(out_w * out_h * out_c);
    part_data.resize(out_w * m_h * m_c);

    float w_scale = static_cast<float>(m_w - 1) / (out_w - 1);
    float h_scale = static_cast<float>(m_h - 1) / (out_h - 1);
    for (int k = 0; k < m_c; ++k) {
        for (int r = 0; r < m_h; ++r) {
            for (int c = 0; c < out_w; ++c) {
                float val = 0;
                if (c == out_w - 1 || m_w == 1) {
                    val = get_pixel(temp_data.get(), m_w - 1, r, k, m_h, m_w, m_c);
                } else {
                    float sx = c * w_scale;
                    int ix = static_cast<int>(sx);
                    float dx = sx - ix;
                    val = (1 - dx) * get_pixel(temp_data.get(), ix, r, k, m_h, m_w, m_c) + dx * get_pixel(temp_data.get(), ix + 1, r, k, m_h, m_w, m_c);
                }
                set_pixel(part_data.get(), c, r, k, val, m_h, out_w, m_c);
            }
        }
    }
    for (int k = 0; k < m_c; ++k) {
        for (int r = 0; r < out_h; ++r) {
            float sy = r * h_scale;
            int iy = static_cast<int>(sy);
            float dy = sy - iy;
            for (int c = 0; c < out_w; ++c) {
                float val = (1 - dy) * get_pixel(part_data.get(), c, iy, k, m_h, out_w, m_c);
                set_pixel(m_data.get(), c, r, k, val, out_h, out_w, out_c);
            }
            if (r == out_h - 1 || m_h == 1)
                continue;
            for (int c = 0; c < out_w; ++c) {
                float val = dy * get_pixel(part_data.get(), c, iy + 1, k, m_h, out_w, m_c);
                add_pixel(m_data.get(), c, r, k, val, out_h, out_w, out_c);
            }
        }
    }

    m_h = out_h;
    m_w = out_w;
    m_c = out_c;
}

void
Image::save(const std::string& filename)
{
    Ptr<unsigned char> tmp_data(m_h * m_w * m_c);
    for (int k = 0; k < m_c; ++k) {
        for (int i = 0; i < m_w * m_h; ++i) {
            tmp_data[i * m_c + k] = static_cast<unsigned char>(255 * m_data[i + k * m_w * m_h]);
        }
    }
    int success = stbi_write_png(filename.c_str(), m_w, m_h, m_c, tmp_data.get(), m_w * m_c);
    if (success)
        printf("Successfully write image to %s\n", filename.c_str());
}

#ifdef DARKNET_OPENCV
void
Image::show(const std::string& window_name)
{
    cv::Mat disp = to_mat();

    cv::namedWindow(window_name, cv::WINDOW_AUTOSIZE);
    cv::imshow(window_name, disp);

    cv::waitKey(0);
}
#endif

/**************
 * Detections *
 **************/

void
Image::draw_detections(const int& num, const float& thresh, box* boxes, float** probs, const std::vector<std::string>& label_list, const int& classes)
{
    std::vector<bbox> bboxes;
    find_bbox(num, thresh, boxes, probs, classes, bboxes);
    for (auto bbox : bboxes) {
        printf("- %s: %.0f%%\n", label_list[bbox.m_label].c_str(), bbox.m_prob * 100);

        int box_width = m_h * 0.012;
        int offset = bbox.m_label * 123457 % classes;
        float red = get_color(2, offset, classes);
        float green = get_color(1, offset, classes);
        float blue = get_color(0, offset, classes);

        draw_box_width(bbox.m_left, bbox.m_top, bbox.m_right, bbox.m_bottom, box_width, red, green, blue);
    }
}

void
Image::find_bbox(const int& num, const float& thresh, box* boxes, float** probs, const int& classes, std::vector<bbox>& ball_position)
{
    // std::vector<bbox> bboxes;

    for (int i = 0; i < num; ++i) {
        int max_id = max_index(probs[i], probs[i] + classes);
        if (probs[i][max_id] > thresh) {
            int left = (boxes[i].x - boxes[i].w / 2.) * m_w;
            int right = (boxes[i].x + boxes[i].w / 2.) * m_w;
            int top = (boxes[i].y - boxes[i].h / 2.) * m_h;
            int bottom = (boxes[i].y + boxes[i].h / 2.) * m_h;

            if (left < 0)
                left = 0;
            if (right > m_w - 1)
                right = m_w - 1;
            if (top < 0)
                top = 0;
            if (bottom > m_h - 1)
                bottom = m_h - 1;

            // std::cout << "left:" << left << std::endl;
            // std::cout << "right:" << right << std::endl;
            // std::cout << "top:" << top << std::endl;
            // std::cout << "bottom:" << bottom << std::endl;

            ball_position.emplace_back(max_id, probs[i][max_id], left, top, right, bottom);
        }
    }
}

void
Image::find_relative_bbox(const int& num, const float& thresh, box* boxes, float** probs, const int& classes, std::vector<RelativeBBox>& ball_position)
{

    for (int i = 0; i < num; ++i) {
        int max_id = max_index(probs[i], probs[i] + classes);
        if (probs[i][max_id] > thresh) {
            ball_position.emplace_back(max_id, probs[i][max_id], boxes[i].x, boxes[i].y, boxes[i].h, boxes[i].w);
        }
    }
}

/***********
 * Getters *
 ***********/

Ptr<float>&
Image::data()
{
    return m_data;
}

#ifdef DARKNET_OPENCV
cv::Mat
Image::to_mat()
{
    Ptr<unsigned char> mat_data(m_h * m_w * m_c);
    cv::Mat disp_bgr(m_h, m_w, CV_8UC3);

    for (int y = 0; y < m_h; ++y) {
        for (int x = 0; x < m_w; ++x) {
            for (int k = 0; k < m_c; ++k) {
                mat_data[y * disp_bgr.step[0] + x * m_c + k] = static_cast<unsigned char>(get_pixel(m_data.get(), x, y, k, m_h, m_w, m_c) * 255);
            }
        }
    }

    cv::Mat disp(m_h, m_w, CV_8UC3, mat_data.get());
    cv::cvtColor(disp, disp_bgr, cv::COLOR_RGB2BGR);

    return disp_bgr;
}

/**********
 * Setter *
 **********/

void
Image::from_mat(const cv::Mat& frame)
{
    m_w = frame.cols;
    m_h = frame.rows;
    m_c = frame.channels();

    cv::Mat converted;
    cv::cvtColor(frame, converted, CV_BGR2RGB);

    if (m_data.size() != static_cast<size_t>(m_h * m_w * m_c)) {
        m_data.resize(m_h * m_w * m_c);
    }

    cv::Mat channels[3];
    std::vector<cv::Mat> channels_vec;
    cv::split(frame, channels);
    channels_vec.push_back(channels[0].reshape(1, 1));
    channels_vec.push_back(channels[1].reshape(1, 1));
    channels_vec.push_back(channels[2].reshape(1, 1));

    cv::Mat dst, dst_2;
    cv::hconcat(channels_vec, dst);
    dst.convertTo(dst_2, CV_32F);
    dst_2 /= 255.;

    // std::cout << "dst:" << static_cast<float>(dst_2.data[0]) << std::endl;
    std::memcpy(m_data.get(), (float*)dst_2.data, sizeof(float) * dst.cols * dst.rows * dst.channels());
    // rgbgr_image(m_data.get());
}

#endif

void
Image::from_array(const uint8_t* frame)
{
    // m_data.resize(m_h * m_w * m_c);

    for (int k = 0; k < m_c; ++k) {
        for (int j = 0; j < m_h; ++j) {
            for (int i = 0; i < m_w; ++i) {
                int dst_index = i + m_w * j + m_w * m_h * k;
                int src_index = k + m_c * i + m_c * m_w * j;
                m_data[dst_index] = static_cast<float>(frame[src_index]) / 255.;
            }
        }
    }
}

/********************
 * Pixel operations *
 ********************/

float
Image::get_pixel(float* data, const int& x, const int& y, const int& c, const int& im_h, const int& im_w, const int& im_c)
{
    assert(x < im_w && y < im_h && c < im_c);
    return data[c * im_h * im_w + y * im_w + x];
}

void
Image::set_pixel(float* data, const int& x, const int& y, const int& c, const float& val, const int& im_h, const int& im_w, const int& im_c)
{
    if (x < 0 || y < 0 || c < 0 || x >= im_w || y >= im_h || c >= im_c)
        return;
    assert(x < im_w && y < im_h && c < im_c);
    data[c * im_h * im_w + y * im_w + x] = val;
}

void
Image::add_pixel(float* data, const int& x, const int& y, const int& c, const float& val, const int& im_h, const int& im_w, const int& im_c)
{
    assert(x < im_w && y < im_h && c < im_c);
    data[c * im_h * im_w + y * im_w + x] += val;
}

/**************
 * Draw boxes *
 **************/

void
Image::draw_box_width(int x1, int y1, int x2, int y2, const int& width, const float& r, const float& g, const float& b)
{
    for (int i = 0; i < width; ++i) {
        draw_box(x1 + i, y1 + i, x2 - i, y2 - i, r, g, b);
    }
}

void
Image::draw_box(int x1, int y1, int x2, int y2, const float& r, const float& g, const float& b)
{
    if (x1 < 0)
        x1 = 0;
    if (x1 >= m_w)
        x1 = m_w - 1;
    if (x2 < 0)
        x2 = 0;
    if (x2 >= m_w)
        x2 = m_w - 1;

    if (y1 < 0)
        y1 = 0;
    if (y1 >= m_h)
        y1 = m_h - 1;
    if (y2 < 0)
        y2 = 0;
    if (y2 >= m_h)
        y2 = m_h - 1;

    for (int i = x1; i <= x2; ++i) {
        m_data[i + y1 * m_w + 0 * m_w * m_h] = r;
        m_data[i + y2 * m_w + 0 * m_w * m_h] = r;

        m_data[i + y1 * m_w + 1 * m_w * m_h] = g;
        m_data[i + y2 * m_w + 1 * m_w * m_h] = g;

        m_data[i + y1 * m_w + 2 * m_w * m_h] = b;
        m_data[i + y2 * m_w + 2 * m_w * m_h] = b;
    }
    for (int i = y1; i <= y2; ++i) {
        m_data[x1 + i * m_w + 0 * m_w * m_h] = r;
        m_data[x2 + i * m_w + 0 * m_w * m_h] = r;

        m_data[x1 + i * m_w + 1 * m_w * m_h] = g;
        m_data[x2 + i * m_w + 1 * m_w * m_h] = g;

        m_data[x1 + i * m_w + 2 * m_w * m_h] = b;
        m_data[x2 + i * m_w + 2 * m_w * m_h] = b;
    }
}

float
Image::get_color(const int& c, const int& x, const int& max)
{
    float colors[6][3] = { { 1, 0, 1 }, { 0, 0, 1 }, { 0, 1, 1 }, { 0, 1, 0 }, { 1, 1, 0 }, { 1, 0, 0 } };
    float ratio = (static_cast<float>(x) / max) * 5;
    int i = floor(ratio);
    int j = ceil(ratio);
    ratio -= i;
    float color = (1 - ratio) * colors[i][c] + ratio * colors[j][c];
    // printf("%f\n", r);
    return color;
}

// void Image::rgbgr_image(float *data) {
//   for (int i = 0; i < m_w * m_h; ++i) {
//     float swap = data[i];
//     data[i] = data[i + m_w * m_h * 2];
//     data[i + m_w * m_h * 2] = swap;
//   }
// }
//
// void Image::constrain_image(float *data) {
//   for (int i = 0; i < m_w * m_h * m_c; ++i) {
//     if (data[i] < 0)
//       data[i] = 0;
//     if (data[i] > 1)
//       data[i] = 1;
//   }
// }

} // namespace darknet
