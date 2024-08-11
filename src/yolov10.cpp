#include "yolov10.h"

YOLOv10::YOLOv10(int input_width, int input_height) 
    : input_width_(input_width), input_height_(input_height) {}

std::vector<std::string> YOLOv10::read_label_names(const std::string& file_name) {
    std::vector<std::string> classes;
    std::ifstream ifs(file_name);
    std::string line;

    while (std::getline(ifs, line)) {
        classes.push_back(line);
    }

    return classes;
}

std::vector<Detection> YOLOv10::postprocess(const cv::Size& frame_size, 
                                             std::vector<std::vector<float>>& infer_results, 
                                             const std::vector<std::vector<int64_t>>& infer_shapes) {
    std::vector<Detection> detections;
    const float confidence_threshold = 0.1f; 
    const auto infer_shape = infer_shapes.front();
    const auto infer_result = infer_results.front();

    int rows = infer_shape[1]; // Assuming this is the number of detections

    for (int i = 0; i < rows; ++i) {
        if (i * infer_shape[2] + 4 >= infer_result.size()) {
            break;
        }

        float score = infer_result[i * infer_shape[2] + 4];
        if (score >= confidence_threshold) {
            Detection detection;
            detection.class_id = static_cast<int>(infer_result[i * infer_shape[2] + 5]);
            detection.class_confidence = score;

            float r_w = static_cast<float>(frame_size.width) / input_width_;
            float r_h = static_cast<float>(frame_size.height) / input_height_;

            detection.bbox.x1 = infer_result[i * infer_shape[2] + 0] * r_w;
            detection.bbox.y1 = infer_result[i * infer_shape[2] + 1] * r_h;
            detection.bbox.x2 = infer_result[i * infer_shape[2] + 2] * r_w;
            detection.bbox.y2 = infer_result[i * infer_shape[2] + 3] * r_h;

            detections.emplace_back(detection);
        }
    }
    return detections; 
}

std::vector<uint8_t> YOLOv10::preprocess(const cv::Mat& img, const std::string& format, 
                                          int img_type1, int img_type3,
                                          size_t img_channels, const cv::Size& img_size) {
    std::vector<uint8_t> input_data;
    cv::Mat sample;
    
    cv::cvtColor(img, sample, cv::COLOR_BGR2RGB); // Convert BGR to RGB
    sample.convertTo(sample, (img_channels == 3) ? img_type3 : img_type1);
    cv::resize(sample, sample, cv::Size(input_width_, input_height_));
    sample.convertTo(sample, CV_32FC3, 1.f / 255.f); // Normalize to [0, 1]

    size_t img_byte_size = sample.total() * sample.elemSize();
    input_data.resize(img_byte_size);

    std::vector<cv::Mat> input_channels;
    size_t pos = 0;

    for (size_t i = 0; i < img_channels; ++i) {
        input_channels.emplace_back(img_size.height, img_size.width, img_type1, &input_data[pos]);
        pos += input_channels.back().total() * input_channels.back().elemSize();
    }

    cv::split(sample, input_channels);

    if (pos != img_byte_size) {
        std::cerr << "Unexpected total size of channels: " << pos << ", expecting " << img_byte_size << std::endl;
        exit(1);
    }

    return input_data;
}
