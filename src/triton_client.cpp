#include "triton_client.h"
#include <stdexcept>
#include <sstream>

static size_t write_callback(char* ptr, size_t size, size_t nmemb, std::string& data) {
    size_t total_size = size * nmemb;
    data.append(ptr, total_size);
    return total_size;
}

TritonModelInfo TritonClient::parse_model(const std::string& model_name, const std::string& url) {
    TritonModelInfo info;
    CURL* curl = curl_easy_init();
    if (!curl) {
        throw std::runtime_error("Failed to initialize libcurl. Ensure that libcurl is installed and properly configured.");
    }
    const auto model_config_url = "http://" + url + ":8000/v2/models/" + model_name + "/config";
    curl_easy_setopt(curl, CURLOPT_URL, model_config_url.c_str());
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_callback);
    std::string response_data;
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response_data);
    CURLcode res = curl_easy_perform(curl);
    if (res != CURLE_OK) {
        curl_easy_cleanup(curl);
        std::stringstream err_msg;
        err_msg << "HTTP request to retrieve model configuration failed. CURL error: " << curl_easy_strerror(res);
        throw std::runtime_error(err_msg.str());
    }
    
    // Unknown model
    if (response_data.find("Request for unknown model") != std::string::npos) {
        curl_easy_cleanup(curl);
        std::stringstream err_msg;
        err_msg << "Model '" << model_name << "' not found on Triton server. Please check the model name and try again.";
        throw std::runtime_error(err_msg.str());
    }

    rapidjson::Document response_json;
    response_json.Parse(response_data.c_str());
    info.input_name = response_json["input"][0]["name"].GetString();
    const auto& input_dims = response_json["input"][0]["dims"].GetArray();
    info.input_format = response_json["input"][0]["format"].GetString();
    if (info.input_format == "FORMAT_NONE") {
        info.input_format = "FORMAT_NCHW";
    }

    if (info.input_format == "FORMAT_NCHW") {
        if (input_dims.Size() == 4) {
            info.input_channels = input_dims[1].GetInt();
            info.input_height = input_dims[2].GetInt();
            info.input_width = input_dims[3].GetInt();
        } else if (input_dims.Size() == 3) {
            info.input_channels = input_dims[0].GetInt();
            info.input_height = input_dims[1].GetInt();
            info.input_width = input_dims[2].GetInt();
            info.input_shape.push_back(info.batch_size);
        } else {
            curl_easy_cleanup(curl);
            std::stringstream err_msg;
            err_msg << "Unsupported input dimensions in model configuration. Expected 3 or 4 dimensions, but received " << input_dims.Size() << " dimensions.";
            throw std::runtime_error(err_msg.str());
        }
    } else if (info.input_format == "FORMAT_NHWC") {
        if (input_dims.Size() == 4) {
            info.input_height = input_dims[1].GetInt();
            info.input_width = input_dims[2].GetInt();
            info.input_channels = input_dims[3].GetInt();
        } else if (input_dims.Size() == 3) {
            info.input_height = input_dims[0].GetInt();
            info.input_width = input_dims[1].GetInt();
            info.input_channels = input_dims[2].GetInt();
            info.input_shape.push_back(info.batch_size);
        } else {
            curl_easy_cleanup(curl);
            std::stringstream err_msg;
            err_msg << "Unsupported input dimensions in model configuration. Expected 3 or 4 dimensions, but received " << input_dims.Size() << " dimensions.";
            throw std::runtime_error(err_msg.str());
        }
    }

    for (const auto& dim : input_dims) {
        info.input_shape.push_back(dim.GetInt64());
    }

    info.max_batch_size = response_json["max_batch_size"].GetInt();

    for (const auto& output : response_json["output"].GetArray()) {
        info.output_names.push_back(output["name"].GetString());
    }

    info.input_datatype = response_json["input"][0]["data_type"].GetString();
    info.input_datatype.erase(0, 5);

    // Cleanup
    curl_easy_cleanup(curl);

    return info;
}

void TritonClient::set_input_shape(const std::vector<int64_t>& shape)
    {
        model_info_.input_shape = shape;
        model_info_.input_channels = shape[1];
        model_info_.input_width = shape[2];
        model_info_.input_height = shape[3];
    }


TritonModelInfo TritonClient::retrieve_model_info(const std::string& model_name, const std::string& url, const std::vector<int64_t>& shape) {
    model_info_ = parse_model(model_name, url);

    if (model_info_.input_width == -1 || model_info_.input_height == -1) {
        if (shape.empty()) {
            throw std::runtime_error("The model has dynamic input sizes. Please provide the input shape.");
        }
        set_input_shape(shape);
    }

    const auto& info = model_info_;
    // Print model info
    // std::cout << "Retrieved model information: " << std::endl;
    // std::cout << "Input Name: " << info.input_name << std::endl;
    // std::cout << "Input Data Type: " << info.input_datatype << std::endl;
    // std::cout << "Input Channels: " << info.input_channels << std::endl;
    // std::cout << "Input Height: " << info.input_height << std::endl;
    // std::cout << "Input Width: " << info.input_width << std::endl;
    // std::cout << "Input Format: " << info.input_format << std::endl;
    // std::cout << "Max Batch Size: " << info.max_batch_size << std::endl;
    // std::cout << "Output Names: ";
    // for (const auto& output_name : info.output_names) {
    //     std::cout << output_name << " ";
    // }
    // std::cout << std::endl;
    return info;
}

void TritonClient::initialize_triton_client() {
    tc::Error err;
    if (protocol_ == ProtocolType::HTTP) {
        err = tc::InferenceServerHttpClient::Create(&client_.httpClient, server_url_, verbose_);
    } else {
        err = tc::InferenceServerGrpcClient::Create(&client_.grpcClient, server_url_, verbose_);
    }
    if (!err.IsOk()) {
        std::stringstream err_msg;
        err_msg << "Failed to create Triton client for inference. Details: " << err;
        throw std::runtime_error(err_msg.str());
    }
}

std::vector<const tc::InferRequestedOutput*> TritonClient::create_infer_requested_output(const std::vector<std::string>& output_names) {
    std::vector<const tc::InferRequestedOutput*> outputs;
    tc::Error err;
    for (const auto& output_name : output_names) {
        tc::InferRequestedOutput* output;
        err = tc::InferRequestedOutput::Create(&output, output_name);
        if (!err.IsOk()) {
            std::stringstream err_msg;
            err_msg << "Unable to create output for inference request. Output name: " << output_name << ". Error details: " << err;
            throw std::runtime_error(err_msg.str());
        } else {
            outputs.push_back(output);
        }
    }
    return outputs;
}

std::tuple<std::vector<std::vector<float>>, std::vector<std::vector<int64_t>>> TritonClient::extract_inference_results(
    tc::InferResult* result,
    const size_t batch_size,
    const std::vector<std::string>& output_names, 
    const bool batching) {
    
    if (!result->RequestStatus().IsOk()) {
        std::stringstream err_msg;
        err_msg << "Inference request failed. Status: " << result->RequestStatus();
        throw std::runtime_error(err_msg.str());
    }

    std::vector<std::vector<float>> infer_results;
    std::vector<std::vector<int64_t>> infer_shapes;

    float* output_data;
    size_t output_byte_size;
    for (const auto& output_name : output_names) {
        std::vector<int64_t> infer_shape;
        std::vector<float> infer_result;

        result->RawData(output_name, (const uint8_t**)&output_data, &output_byte_size);

        tc::Error err = result->Shape(output_name, &infer_shape);
        infer_result = std::vector<float>(output_byte_size / sizeof(float));
        std::memcpy(infer_result.data(), output_data, output_byte_size);
        if (!err.IsOk()) {
            std::stringstream err_msg;
            err_msg << "Failed to retrieve data for output: " << output_name << ". Error details: " << err;
            throw std::runtime_error(err_msg.str());
        }
        infer_results.push_back(infer_result);
        infer_shapes.push_back(infer_shape);
    }

    return std::make_tuple(infer_results, infer_shapes);
}

std::tuple<std::vector<std::vector<float>>, std::vector<std::vector<int64_t>>> TritonClient::run_inference(const std::vector<uint8_t>& input_data) {
    tc::Error err;
    std::vector<tc::InferInput*> inputs = {nullptr};

    std::string input_name = model_info_.input_name;
    err = tc::InferInput::Create(&inputs[0], input_name, model_info_.input_shape, model_info_.input_datatype);
    if (!err.IsOk()) {
        std::stringstream err_msg;
        err_msg << "Failed to create inference input. Error details: " << err;
        throw std::runtime_error(err_msg.str());
    }

    err = inputs[0]->AppendRaw(input_data);
    if (!err.IsOk()) {
        std::stringstream err_msg;
        err_msg << "Failed to set input data for inference. Error details: " << err;
        throw std::runtime_error(err_msg.str());
    }

    tc::InferOptions options(model_name_);
    std::vector<std::string> output_names = model_info_.output_names;
    auto outputs = create_infer_requested_output(output_names);

    tc::InferResult* result;
    std::unique_ptr<tc::InferResult> result_ptr;
    if (protocol_ == ProtocolType::HTTP) {
        err = client_.httpClient->Infer(&result, options, inputs, outputs);
    } else {
        err = client_.grpcClient->Infer(&result, options, inputs, outputs);
    }

    if (!err.IsOk()) {
        std::stringstream err_msg;
        err_msg << "Inference request failed. Error details: " << err;
        throw std::runtime_error(err_msg.str());
    }
    
    const auto [infer_results, infer_shapes] =  extract_inference_results(result, inputs[0]->Shape()[0], output_names, model_info_.max_batch_size > 0);
    result_ptr.reset(result);
    return std::make_tuple(infer_results, infer_shapes);
}


