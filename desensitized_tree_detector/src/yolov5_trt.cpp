#include "yolov5_trt.h"

#include <NvInferPlugin.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <stdexcept>

class Logger : public nvinfer1::ILogger
{
    void log(Severity severity, const char *msg) noexcept override
    {
        if (severity <= Severity::kWARNING)
        {
            std::cout << "[TensorRT] " << msg << std::endl;
        }
    }
} gLogger;

YOLOv5TRT::YOLOv5TRT(const std::string &engine_path, int dla_core)
{
    initLibNvInferPlugins(&gLogger, "");

    std::ifstream file(engine_path, std::ios::binary);
    if (!file.good())
    {
        throw std::runtime_error("Engine file not found: " + engine_path);
    }

    file.seekg(0, file.end);
    size_t size = static_cast<size_t>(file.tellg());
    file.seekg(0, file.beg);

    std::vector<char> engine_data(size);
    file.read(engine_data.data(), static_cast<std::streamsize>(size));

    runtime = nvinfer1::createInferRuntime(gLogger);
    if (dla_core >= 0)
    {
        runtime->setDLACore(dla_core);
    }

    engine = runtime->deserializeCudaEngine(engine_data.data(), size);
    if (!engine)
    {
        throw std::runtime_error("Failed to deserialize TensorRT engine");
    }

    context = engine->createExecutionContext();
    if (!context)
    {
        throw std::runtime_error("Failed to create TensorRT execution context");
    }

    int nbIOTensors = engine->getNbIOTensors();
    for (int i = 0; i < nbIOTensors; ++i)
    {
        const char *name = engine->getIOTensorName(i);
        nvinfer1::TensorIOMode mode = engine->getTensorIOMode(name);
        if (mode == nvinfer1::TensorIOMode::kINPUT)
        {
            m_input_name = name;
        }
        else if (mode == nvinfer1::TensorIOMode::kOUTPUT)
        {
            m_output_name = name;
        }
    }

    if (m_input_name.empty() || m_output_name.empty())
    {
        throw std::runtime_error("Input or output tensor name not found");
    }

    nvinfer1::Dims input_dims = engine->getTensorShape(m_input_name.c_str());
    if (input_dims.nbDims == 4)
    {
        m_input_c = input_dims.d[1];
        m_input_h = input_dims.d[2];
        m_input_w = input_dims.d[3];
    }

    nvinfer1::Dims output_dims = engine->getTensorShape(m_output_name.c_str());
    if (output_dims.nbDims == 3)
    {
        m_num_detections = output_dims.d[1];
        m_elements_per_det = output_dims.d[2];
        m_num_classes = m_elements_per_det - 5;
    }

    cudaError_t err1 = cudaMalloc(&buffers[0], m_input_c * m_input_h * m_input_w * sizeof(float));
    cudaError_t err2 = cudaMalloc(&buffers[1], m_num_detections * m_elements_per_det * sizeof(float));
    if (err1 != cudaSuccess || err2 != cudaSuccess)
    {
        throw std::runtime_error("Failed to allocate CUDA buffers");
    }

    cudaStreamCreate(&stream);
}

YOLOv5TRT::~YOLOv5TRT()
{
    if (stream)
        cudaStreamDestroy(stream);
    if (buffers[0])
        cudaFree(buffers[0]);
    if (buffers[1])
        cudaFree(buffers[1]);
    if (context)
        delete context;
    if (engine)
        delete engine;
    if (runtime)
        delete runtime;
}

std::vector<Detection> YOLOv5TRT::detect(const cv::Mat &image)
{
    return infer(image);
}

std::vector<Detection> YOLOv5TRT::infer(const cv::Mat &image)
{
    std::vector<Detection> detections;
    preprocess(image, static_cast<float *>(buffers[0]));

    if (!context->setTensorAddress(m_input_name.c_str(), buffers[0]))
    {
        throw std::runtime_error("Failed to bind input tensor address");
    }
    if (!context->setTensorAddress(m_output_name.c_str(), buffers[1]))
    {
        throw std::runtime_error("Failed to bind output tensor address");
    }

    bool status = context->enqueueV3(stream);
    if (!status)
    {
        throw std::runtime_error("TensorRT enqueueV3 failed");
    }

    size_t output_size = static_cast<size_t>(m_num_detections) * static_cast<size_t>(m_elements_per_det);
    std::vector<float> output(output_size);
    cudaMemcpyAsync(output.data(), buffers[1], output.size() * sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    postprocess(output.data(), image.size(), detections);
    return detections;
}

void YOLOv5TRT::preprocess(const cv::Mat &image, float *device_buffer)
{
    if (image.empty())
        return;

    cudaMemset(device_buffer, 0, m_input_c * m_input_h * m_input_w * sizeof(float));

    cv::Mat resized;
    cv::resize(image, resized, cv::Size(m_input_w, m_input_h));

    cv::Mat rgb;
    cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);
    rgb.convertTo(rgb, CV_32FC3, 1.0 / 255.0);

    std::vector<cv::Mat> channels(3);
    cv::split(rgb, channels);

    std::vector<float> host_data(static_cast<size_t>(m_input_c) * static_cast<size_t>(m_input_h) * static_cast<size_t>(m_input_w));
    memcpy(host_data.data(), channels[0].data, m_input_h * m_input_w * sizeof(float));
    memcpy(host_data.data() + m_input_h * m_input_w, channels[1].data, m_input_h * m_input_w * sizeof(float));
    memcpy(host_data.data() + 2 * m_input_h * m_input_w, channels[2].data, m_input_h * m_input_w * sizeof(float));

    cudaMemcpy(device_buffer, host_data.data(), host_data.size() * sizeof(float), cudaMemcpyHostToDevice);
}

void YOLOv5TRT::postprocess(const float *output, const cv::Size &image_size, std::vector<Detection> &detections)
{
    detections.clear();

    const float scale_x = static_cast<float>(image_size.width) / static_cast<float>(m_input_w);
    const float scale_y = static_cast<float>(image_size.height) / static_cast<float>(m_input_h);

    for (int i = 0; i < m_num_detections; ++i)
    {
        const float obj_conf = output[i * m_elements_per_det + 4];
        if (obj_conf < 0.25f)
            continue;

        const float *class_scores = output + i * m_elements_per_det + 5;
        auto best_it = std::max_element(class_scores, class_scores + m_num_classes);
        float conf = obj_conf * (*best_it);
        if (conf < conf_threshold_)
            continue;

        const float cx = output[i * m_elements_per_det + 0];
        const float cy = output[i * m_elements_per_det + 1];
        const float w = output[i * m_elements_per_det + 2];
        const float h = output[i * m_elements_per_det + 3];

        float x1 = (cx - w / 2.0f) * scale_x;
        float y1 = (cy - h / 2.0f) * scale_y;
        float x2 = (cx + w / 2.0f) * scale_x;
        float y2 = (cy + h / 2.0f) * scale_y;

        if (std::isnan(x1) || std::isnan(y1) || std::isnan(x2) || std::isnan(y2))
            continue;

        x1 = std::max(0.0f, x1);
        y1 = std::max(0.0f, y1);
        x2 = std::min(static_cast<float>(image_size.width), x2);
        y2 = std::min(static_cast<float>(image_size.height), y2);

        if (x2 <= x1 || y2 <= y1)
            continue;

        Detection det;
        det.box = cv::Rect(cv::Point(static_cast<int>(x1), static_cast<int>(y1)),
                           cv::Point(static_cast<int>(x2), static_cast<int>(y2)));
        det.conf = conf;
        det.class_id = static_cast<int>(std::distance(class_scores, best_it));
        detections.push_back(det);
    }
}
