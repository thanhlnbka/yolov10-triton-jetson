#pragma once
#include "common.h"
#include <curl/curl.h>
#include <rapidjson/document.h>

struct TritonModelInfo {
    std::string output_name;
    std::vector<std::string> output_names;
    std::string input_name;
    std::string input_datatype;
    int input_channels;    // Input channels (C)
    int input_height;      // Input height (H)
    int input_width;       // Input width (W)
    std::string input_format;
    int single_channel_type{CV_32FC1};
    int three_channel_type{CV_32FC3};
    int max_batch_size;
    int batch_size{1};
    std::vector<int64_t> input_shape;
};

union TritonClientInstance {
    TritonClientInstance() {
        new (&httpClient) std::unique_ptr<tc::InferenceServerHttpClient>{};
    }
    ~TritonClientInstance() {}

    std::unique_ptr<tc::InferenceServerHttpClient> httpClient;
    std::unique_ptr<tc::InferenceServerGrpcClient> grpcClient;
};

enum class ProtocolType { HTTP = 0, GRPC = 1 };

class TritonClient {
private:
    TritonClientInstance client_;
    const std::string& server_url_;
    bool verbose_;
    ProtocolType protocol_;
    std::string model_name_;
    TritonModelInfo model_info_;
    std::string model_version_;

public:
    TritonClient(const std::string& server_url, ProtocolType protocol, const std::string& model_name, const std::string& model_version = "", bool verbose = false)
        : server_url_{server_url},
          verbose_{verbose},
          protocol_{protocol},
          model_name_{model_name},
          model_version_{model_version} {}

    TritonModelInfo parse_model(const std::string& model_name, const std::string& url);
    TritonModelInfo retrieve_model_info(const std::string& model_name, const std::string& url, const std::vector<int64_t>& shape);

    void set_input_shape(const std::vector<int64_t>& shape);

    void initialize_triton_client();
    std::tuple<std::vector<std::vector<float>>, std::vector<std::vector<int64_t>>> run_inference(const std::vector<uint8_t>& input_data);
    std::vector<const tc::InferRequestedOutput*> create_infer_requested_output(const std::vector<std::string>& output_names);
    std::tuple<std::vector<std::vector<float>>, std::vector<std::vector<int64_t>>> extract_inference_results(
        tc::InferResult* result,
        size_t batch_size,
        const std::vector<std::string>& output_names,
        bool batching);
};
