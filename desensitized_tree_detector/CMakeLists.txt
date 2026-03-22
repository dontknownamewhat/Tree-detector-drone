cmake_minimum_required(VERSION 3.12)
project(tree_detector_desensitized LANGUAGES CXX CUDA)

find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)

set(TensorRT_INCLUDE_DIRS "/usr/include")
set(TensorRT_LIBRARIES nvinfer nvinfer_plugin)

add_library(tree_detector_desensitized SHARED
    src/yolov5_trt.cpp
    src/tree_detector.cpp
)

target_include_directories(tree_detector_desensitized
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
    ${TensorRT_INCLUDE_DIRS}
)

target_link_libraries(tree_detector_desensitized
    ${OpenCV_LIBS}
    ${CUDA_LIBRARIES}
    ${TensorRT_LIBRARIES}
    cudart
    nlohmann_json::nlohmann_json
)

# 脱敏版本说明：
# 1) 移除了组织内部 include 路径和内部库依赖。
# 2) 移除了业务数据库、设备ID、照片落盘等系统耦合代码。
# 3) 保留 tree detector 核心流程：2D检测 + 3D估计 + 树冠凸出 + 净空估算。
