#pragma once

#include <NvInfer.h>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

struct Detection
{
    cv::Rect box;
    float conf;
    int class_id;
    Detection() : box(cv::Rect()), conf(0.0f), class_id(-1) {}
    Detection(cv::Rect b, float c, int id) : box(b), conf(c), class_id(id) {}
};

class YOLOv5TRT
{
public:
    explicit YOLOv5TRT(const std::string &engine_path, int dla_core = -1);
    ~YOLOv5TRT();

    std::vector<Detection> detect(const cv::Mat &image);
    std::vector<Detection> infer(const cv::Mat &image);

    void setConfThreshold(float conf) { conf_threshold_ = conf; }
    void setNMSThreshold(float nms) { nms_threshold_ = nms; }

private:
    void preprocess(const cv::Mat &image, float *device_buffer);
    void postprocess(const float *output, const cv::Size &image_size, std::vector<Detection> &detections);

    nvinfer1::IRuntime *runtime{nullptr};
    nvinfer1::ICudaEngine *engine{nullptr};
    nvinfer1::IExecutionContext *context{nullptr};

    void *buffers[2]{nullptr, nullptr};
    cudaStream_t stream{};

    float conf_threshold_ = 0.5f;
    float nms_threshold_ = 0.45f;

    int m_input_c = 3;
    int m_input_h = 640;
    int m_input_w = 640;
    int m_num_detections = 25200;
    int m_num_classes = 1;
    int m_elements_per_det = 6;

    std::string m_input_name;
    std::string m_output_name;
};
