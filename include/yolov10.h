#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <fstream>
#include <variant>
#include "common.h"

struct Box {
    float x1, y1, x2, y2;
};

struct Detection {
    // Detection-specific fields
    Box bbox;
    float class_id;
    float class_confidence;
};


class YOLOv10 {
public:
    YOLOv10(int input_width, int input_height);

    std::vector<Detection> postprocess(const cv::Size& frame_size, std::vector<std::vector<float>>& infer_results,
                                    const std::vector<std::vector<int64_t>>& infer_shapes);

    std::vector<uint8_t> preprocess(const cv::Mat& img, const std::string& format, int img_type1, int img_type3,
                                    size_t img_channels, const cv::Size& img_size);
    
    std::vector<std::string> read_label_names(const std::string& file_name);

private:
    int input_width_;
    int input_height_;
};
