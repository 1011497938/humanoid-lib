/**
 * @Author: Yusu Pan <yuthon>
 * @Date:   2017-03-18T12:59:40+08:00
 * @Email:  xxdsox@gmail.com
 * @Project: Dancer2017
 * @Filename: darknet.cpp
 * @Last modified by:   yuthon
 * @Last modified time: 2017-03-21T16:31:08+08:00
 * @Copyright: ZJUDancer
 */

#include "darknetcxx/detector.hpp"
#include "darknetcxx/utils.hpp"
#include <string>

using namespace darknet;

void
usage()
{
    printf("./darknet detector [test/demo]\n"
           "\t--data path/to/data_cfg\n"
           "\t--net path/to/net_cfg\n"
           "\t--weights path/to/weights\n"
           "\t[--image path/to/image]\n"
           "\t[--cam cam_index]\n"
           "\t[--video /path/to/video]\n"
           "\t[-thresh threshold]\n");
    exit(1);
}

int
main(int argc, char const* argv[])
{
    InputParser args(argc, argv);
    // run detector
    if (args.cmdOptionExists("detector")) {
        const std::string& mode = args.getCmdOption("detector");
        const std::string data_cfg = fix_path(args.getCmdOption("--data"));
        const std::string net_cfg = fix_path(args.getCmdOption("--net"));
        const std::string weight_file = fix_path(args.getCmdOption("--weights"));
        const float thresh = args.getCmdOption("--thresh") == "" ? 0.1 : std::stof(args.getCmdOption("--thresh"));
        const std::string image_file = fix_path(args.getCmdOption("--image"));
        const int cam_index = args.getCmdOption("--cam") == "" ? 0 : std::stoi(args.getCmdOption("--cam"));
        const std::string video_file = args.getCmdOption("--video") == "" ? "" : fix_path(args.getCmdOption("--video"));

        if (data_cfg == "" || net_cfg == "" || weight_file == "") {
            usage();
        } else {
            // run test
            if (mode == "test" && image_file != "") {
                test_detector(data_cfg, net_cfg, weight_file, image_file, thresh);
#ifdef DARKNET_OPENCV
                // run demo
            } else if (mode == "demo") {
                demo_detector(data_cfg, net_cfg, weight_file, cam_index, video_file, thresh);
#endif
            } else {
                usage();
            }
        }
    } else {
        usage();
    }
    return 0;
}
