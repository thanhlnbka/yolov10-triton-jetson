#include <iostream>
#include "yolov10.h"
#include "triton_client.h"

void draw(cv::Mat& image, const std::string& label, float conf, int left, int top) {
    const float FONT_SCALE = 0.5;
    const int THICKNESS = 1;
    cv::Scalar TEXT_COLOR(255, 255, 255);
    cv::Scalar BACKGROUND_COLOR(0, 0, 0);
    std::string display_text = label + ": " + std::to_string(conf);
    int baseLine;
    cv::Size label_size = cv::getTextSize(display_text, cv::FONT_HERSHEY_DUPLEX, FONT_SCALE, THICKNESS, &baseLine);
    top = std::max(top, label_size.height);
    
    cv::Point tlc(left, top);
    cv::Point brc(left + label_size.width, top + label_size.height + baseLine);
    
    cv::rectangle(image, tlc, brc, BACKGROUND_COLOR, cv::FILLED);
    cv::putText(image, display_text, {left, top + label_size.height}, cv::FONT_HERSHEY_DUPLEX, FONT_SCALE, TEXT_COLOR, THICKNESS);
}

std::vector<Detection> infer_image(const cv::Mat& source, 
    const std::unique_ptr<YOLOv10>& task, 
    const std::unique_ptr<TritonClient>& tritonClient, 
    const TritonModelInfo& modelInfo) {
    
    auto start_pre = std::chrono::steady_clock::now();
    std::vector<uint8_t> input_data = task->preprocess(source, modelInfo.input_format, modelInfo.single_channel_type, 
                                                        modelInfo.three_channel_type, modelInfo.input_channels, 
                                                        cv::Size(modelInfo.input_width, modelInfo.input_height));
    
    auto end_pre = std::chrono::steady_clock::now();
    auto diff_pre = std::chrono::duration_cast<std::chrono::milliseconds>(end_pre - start_pre).count();
    std::cout << "Preprocess time: " << diff_pre << " ms" << std::endl;
    auto start_infer = std::chrono::steady_clock::now();
    auto [infer_results, infer_shapes] = tritonClient->run_inference(input_data);

    auto end_infer = std::chrono::steady_clock::now();
    auto diff_infer = std::chrono::duration_cast<std::chrono::milliseconds>(end_infer - start_infer).count();
    std::cout << "Infer time: " << diff_infer << " ms" << std::endl;
    
    return task->postprocess(cv::Size(source.cols, source.rows), infer_results, infer_shapes);
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << "<path_to_image>" << std::endl;
        return 1; 
    }

    std::cout << "START TRITON CLIENT" << std::endl;
    std::vector<int64_t> input_sizes{1, 3, 640, 640}; 
    std::string server_address = "localhost";
    std::string model_name = "yolov10m";
    std::string url = server_address + ":8001"; // 8000 HTTP
    ProtocolType protocol = ProtocolType::GRPC; // ProtocolType::HTTP

    // Create Triton client
    std::unique_ptr<TritonClient> tritonClient = std::make_unique<TritonClient>(url, protocol, model_name);
    tritonClient->initialize_triton_client();

    TritonModelInfo modelInfo = tritonClient->retrieve_model_info(model_name, server_address, input_sizes);
    std::unique_ptr<YOLOv10> task = std::make_unique<YOLOv10>(modelInfo.input_width, modelInfo.input_height);
    const auto class_names = task->read_label_names("../labels/classes.txt");
    auto start = std::chrono::steady_clock::now();
    std::string image_path = argv[1]; 
    cv::Mat image = cv::imread(image_path);
    if (image.empty()) {
        std::cerr << "Error loading image: " << image_path << std::endl;
        return 1; 
    }
    std::vector<Detection> predictions = infer_image(image, task, tritonClient, modelInfo);
    auto end = std::chrono::steady_clock::now();
    auto diff = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    
    for (const Detection& detection : predictions) {
        cv::Rect bbox = cv::Rect(cv::Point(detection.bbox.x1, detection.bbox.y1), cv::Point(detection.bbox.x2, detection.bbox.y2));
        cv::rectangle(image, bbox, cv::Scalar(255, 0, 0), 2);
        draw(image, class_names[detection.class_id], detection.class_confidence, bbox.x, bbox.y - 1);
    }    
    std::cout << "Total time: " << diff << " ms" << std::endl;
    std::string processedFrameFilename = "processed_image.jpg";
    std::cout << "Saving image processed: " << processedFrameFilename << std::endl;
    cv::imwrite(processedFrameFilename, image);
    return 0;
}
