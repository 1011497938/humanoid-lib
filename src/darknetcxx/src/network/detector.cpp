/**
 * @Author: Yusu Pan <yuthon>
 * @Date:   2017-03-18T20:13:33+08:00
 * @Email:  xxdsox@gmail.com
 * @Project: Dancer2017
 * @Filename: detector.cpp
 * @Last modified by:   yuthon
 * @Last modified time: 2017-03-18T22:21:17+08:00
 * @Copyright: ZJUDancer
 */

#include "darknetcxx/detector.hpp"
#include <algorithm>
#include <map>
#ifdef DARKNET_OPENCV
#include <opencv2/opencv.hpp>
#endif
#include <ctime>
#include <string>
#include <sys/time.h>
#include <vector>

namespace darknet {
static float fps = 0;

void
test_detector(const std::string& data_cfg, const std::string& net_cfg, const std::string& weight_file, const std::string& image_file, const float& thresh, const float& higher_thresh)
{
    // parse data cfg && read label name list
    std::vector<std::string> label_list = read_data_cfg(data_cfg);

    // parse net cfg and setup network
    Network* net = &parse_network_cfg(net_cfg);
    params p = net->get_params();
    printf("Setup: num_layers = %d, batch = %d\n", net->num_layers(), p.batch);

    // load pretrained weights
    // if (weight_file.empty()) {
    //   LOG(ERROR, "weigth file doesn't exist!");
    // }
    load_weights(net, weight_file);

    // set batch to 1 for network inference
    net->set_network_batch(1);

    // read image data and resize
    // if (!image_file.empty()) {
    Image im(image_file);
    // log_layer_output(im.data().get(), im.data().size(), 100);
    im.resize_neo(p.h, p.w, p.c);
    // log_layer_output(im.data().get(), im.data().size(), 101);
    // }

    // inference and draw bbox
    bbox_detection(net, &im, label_list, thresh);

    im.save("predicted.png");
#ifdef DARKNET_OPENCV
#ifndef DARKNET_NO_DISPLAY
    im.show();
#endif
#endif
}

#ifdef DARKNET_OPENCV
void
demo_detector(const std::string& data_cfg,
              const std::string& net_cfg,
              const std::string& weight_file,
              const int& cam_index,
              const std::string& video_file,
              const float& thresh,
              const float& higher_thresh,
              const int& frame_skip,
              const std::string& prefix)
{
    // parse data cfg && read label name list
    std::vector<std::string> label_list = read_data_cfg(data_cfg);

    // parse net cfg and setup network
    Network* net = &parse_network_cfg(net_cfg);
    params p = net->get_params();
    printf("Setup: num_layers = %d, batch = %d\n", net->num_layers(), p.batch);

    // load pretrained weights
    // if (weight_file.empty()) {
    //   LOG(ERROR, "weigth file doesn't exist!");
    // }
    load_weights(net, weight_file);

    // set batch to 1 for network inference
    net->set_network_batch(1);

    // read frame from video file or camera
    cv::VideoCapture cap;
    cv::Mat frame;
    cv::Mat frame_result;
    cv::Size size(p.w, p.h);
    Image im(p.w, p.h, 3, false);

    if (video_file != "") {
        printf("capture from video file: %s\n", video_file.c_str());
        cap.open(video_file);
    } else {
        cap.open(cam_index);
    }
    // if (!cap.isOpened) {
    //   LOG(ERROR, "Couldn't connect to webcam.");
    // }

    cv::namedWindow("demo", cv::WINDOW_AUTOSIZE);
    for (;;) {
        clock_t time;
        cap >> frame; // get a new frame from camera
        cv::resize(frame, frame_result, size);
        im.from_mat(frame_result);

        // inference and draw bbox
        time = clock();
        bbox_detection(net, &im, label_list, thresh);
        fps = 1. / sec(clock() - time);

        frame_result = im.to_mat();
        cv::imshow("demo", frame_result);
        if (cv::waitKey(30) >= 0)
            break;
    }
}
#endif

void
bbox_detection(Network* net, Image* im, const std::vector<std::string>& label_list, const float& thresh, const bool verbose)
{
#ifdef DARKNET_GPU
    net->network_predict_gpu(im->data());
#else
    net->network_predict(im->data());
#endif
    // get predicted boxes and their probs
    Detection* detection = dynamic_cast<Detection*>(net->get_layers().back());
    std::map<std::string, int> d_p = detection->get_params();
    const int side = d_p["side"];
    const int n = d_p["n"];
    const int classes = d_p["classes"];
    // FIXME memory leak: boxes && probs
    box* boxes = reinterpret_cast<box*>(calloc(side * side * n, sizeof(box)));
    float** probs = new float*[side * side * n];
    for (int j = 0; j < side * side * n; ++j)
        probs[j] = new float[classes];
    detection->get_detection_boxes(1, 1, thresh, probs, boxes, 0);

    box tmp_box(0, 0, 0, 0);
    tmp_box.nms_sort(boxes, probs, side * side * n, classes, thresh);

    if (verbose) {
        printf("\033[2J");
        printf("\033[1;1H");
        printf("\nFPS:%.1f\n", fps);
        printf("Objects:\n");
    }

    // output detections
    im->draw_detections(side * side * n, thresh, boxes, probs, label_list, classes);
    free(boxes);
    free_ptrs(reinterpret_cast<void**>(probs), side * side * n);
}

void
obj_detection(Network* net, Image* im, const float& thresh, std::vector<RelativeBBox>& ball_position)
{
#ifdef DARKNET_GPU
    net->network_predict_gpu(im->data());
#else
    net->network_predict(im->data());
#endif

    // get predicted boxes and their probs
    Detection* detection = dynamic_cast<Detection*>(net->get_layers().back());
    std::map<std::string, int> d_p = detection->get_params();
    const int side = d_p["side"];
    const int n = d_p["n"];
    const int classes = d_p["classes"];

    // FIXME memory leak: boxes && probs
    box* boxes = reinterpret_cast<box*>(calloc(side * side * n, sizeof(box)));
    float** probs = new float*[side * side * n];
    for (int j = 0; j < side * side * n; ++j)
        probs[j] = new float[classes];

    detection->get_detection_boxes(1, 1, thresh, probs, boxes, 0);

    box tmp_box(0, 0, 0, 0);
    tmp_box.nms_sort(boxes, probs, side * side * n, classes, thresh);

    // output detections
    im->find_relative_bbox(side * side * n, thresh, boxes, probs, classes, ball_position);
}
} // namespace darknet
