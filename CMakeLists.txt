cmake_minimum_required(VERSION 3.10)
project(triton-client)
set(CMAKE_CXX_STANDARD 20)

# Define the path to the Triton libraries and CMake modules
set(CLIENTS_DIR "${PROJECT_SOURCE_DIR}/clients")
set(TritonClient_DIR "${CLIENTS_DIR}/lib/cmake/TritonClient")
set(TritonCommon_DIR "${CLIENTS_DIR}/lib/cmake/TritonCommon")

find_package(OpenCV REQUIRED)
find_package(TritonCommon REQUIRED)
find_package(TritonClient REQUIRED)
find_package(CURL REQUIRED)
find_package(RapidJSON REQUIRED)

# Define source files
set(SOURCES 
    ${PROJECT_SOURCE_DIR}/src/main.cpp
    ${PROJECT_SOURCE_DIR}/src/triton_client.cpp
    ${PROJECT_SOURCE_DIR}/src/yolov10.cpp

)

add_executable(${PROJECT_NAME} ${SOURCES})

# Include directories
target_include_directories(${PROJECT_NAME} PRIVATE
    ${OpenCV_INCLUDE_DIRS}
    ${CLIENTS_DIR}/include
    ${PROJECT_SOURCE_DIR}/include
)

# Link directories and libraries
target_link_directories(${PROJECT_NAME} PRIVATE ${CLIENTS_DIR}/lib)

target_link_libraries(${PROJECT_NAME} PRIVATE
    grpcclient
    httpclient
    ${OpenCV_LIBS}
    CURL::libcurl
)
