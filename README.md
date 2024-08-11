# YOLOv10 Triton Jetson

## Introduction

This project provides guidance on exporting the YOLOv10 model from PyTorch to ONNX, converting it to a TensorRT engine, and deploying it on the NVIDIA Triton Inference Server on a Jetson device with JetPack 5.1.3

## Instructions

1. Clone the Project and Set Up the Environment

    First, clone the project repository:
    ```bash
    git clone https://github.com/thanhlnbka/yolov10-triton-jetson.git
    ```
    Next, download and extract the Triton server for JetPack:
    ```bash
    wget https://github.com/triton-inference-server/server/releases/download/v2.34.0/tritonserver2.34.0-jetpack5.1.tgz
    tar -xvzf tritonserver2.34.0-jetpack5.1.tgz
    cp -r clients yolov10-triton-jetson/.
    cp -r tritonserver yolov10-triton-jetson/servers/.
    ```

## I. Set Up the Triton Server on Jetson
### Export YOLOv10 Model from PyTorch to ONNX

1. **Build env-yolov10**:
   ```bash
   git clone https://github.com/THU-MIG/yolov10.git
   cd yolov10/docker
   docker build -t yolov10 -f Dockerfile-jetson --name env-yolov10 .
   ```
2. **Using env-yolov10 for export ONNX**
    ```bash
    docker run -it --network host --gpus all --runtime nvidia --name env-yolov10 yolov10 bash
    yolo export model=jameslahm/yolov10m.pt format=onnx opset=13 simplify
    ```

3. **Copy the Model to the Woring Directory**
    ```bash
    docker cp env-yolov10:/usr/src/ultralytics/jameslahm yolov10-triton-jetson/servers/models
    ```

### Convert ONNX Model to TensorRT Engine

1. Run the TensorRT Docker Container:
    ```bash
    docker run -it -v yolov10-triton-jetson/servers:/servers --network host --gpus all --runtime nvidia --name env-engine nvcr.io/nvidia/l4t-tensorrt:r8.5.2.2-devel bash
    ```

2. Convert ONNX to TensorRT:
    ```bash
    cd /servers/models
    /usr/src/tensorrt/bin/trtexec --onnx=yolov10m.onnx --saveEngine=yolov10m.engine --fp16 --useCudaGraph
    ```
    
### Deploy YOLOv10 on the Triton Server
    
1. Setup Triton
    ```bash
    docker exec -it env-engine bash
    cd /servers
    sh setup_triton.sh 
    mkdir -p model_repository/yolov10m/1
    cp /servers/models/yolov10m.engine /servers/model_repository/yolov10m/1/model.plan
    ```
    
2. Start Triton
    ```bash
    tritonserver --model-repository=/servers/model_repository  --backend-directory=/work/tritonserver/backends --log-verbose=1
     ```
# II. Set Up the Client to Communicate with Triton

1. Build the Client Source Code:
    ```bash
    cd yolov10-triton-jetson
    mkdir build
    cd build
    cmake .. && make 
    ```
2. Test the Client:
    ```bash
    ./triton-client <path_to_image>
    ```

3. Demo Result:

    ![all_about_people_cover.jpeg](./images/processed_image.jpg)

# References 

* https://github.com/triton-inference-server/server
* https://github.com/THU-MIG/yolov10
    
    