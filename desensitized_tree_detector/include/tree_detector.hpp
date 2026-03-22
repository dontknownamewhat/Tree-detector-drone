#ifndef TREE_DETECTOR_HPP
#define TREE_DETECTOR_HPP

#include <chrono>
#include <memory>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

#include "yolov5_trt.h"

// ===== 脱敏说明 =====
// 本头文件已移除组织内私有模块的真实类型，改为通用接口。
// 对外只保留 tree detector 所需的数据结构与方法签名。

enum class DeviceTypeEnum
{
    VegetationEncroachment = 0
};

struct DeviceDetectResult
{
    int dev_xmin = 0;
    int dev_ymin = 0;
    int dev_xmax = 0;
    int dev_ymax = 0;

    float score = 0.0f;
    int dev_size = 0;

    float dx = 0.0f;
    float dy = 0.0f;
    float dz = 0.0f;

    float dxmin = 0.0f;
    float dxmax = 0.0f; // 在本实现中也用于存储净空距离
    float dymin = 0.0f;
    float dymax = 0.0f;

    int clearance_px = 0;
    int clearance_py = 0;
    bool has_clearance_point = false;

    DeviceDetectResult() = default;
    DeviceDetectResult(int x, int y, int w, int h)
        : dev_xmin(x), dev_ymin(y), dev_xmax(x + w), dev_ymax(y + h)
    {
    }
};

struct MeasureDeviceResult
{
    int time_sample = 0;
    DeviceTypeEnum device_type = DeviceTypeEnum::VegetationEncroachment;
    bool is_detect = false;
    std::vector<DeviceDetectResult> detect_list;
};

struct MissionStatus
{
    int fly_line_type = 0;
};

struct PointCloudData
{
    float dx = 0.0f;
    float dy = 0.0f;
    float dz = 0.0f;
    float piexl_x = 0.0f;
    float piexl_y = 0.0f;
};

struct PointCloudDetectConfig
{
    int detect_start_x = 0;
    int detect_start_y = 0;
    int detect_end_x = 0;
    int detect_end_y = 0;

    int x_land = 2;
    int y_land = 2;
    int y_large_land = 6;
    int threshold = 40;
    float min_score = 0.6f;
};

struct PointCloudBaseMeasureResult
{
    cv::Mat imageLeft;
    cv::Mat imageRight;
    PointCloudDetectConfig config;
};

template <typename T>
class IThreadSafeData
{
public:
    virtual ~IThreadSafeData() = default;
    virtual std::shared_ptr<T> getNextData() = 0;
    virtual std::shared_ptr<T> getCurrentData() = 0;
    virtual void changeToNextData() = 0;
};

class IPointCloudBase
{
public:
    virtual ~IPointCloudBase() = default;
    virtual std::shared_ptr<IThreadSafeData<PointCloudBaseMeasureResult>> getMeasureResult() = 0;
    virtual void calculatePointCloudForList(const cv::Mat &left_gray,
                                            const cv::Mat &right_gray,
                                            const PointCloudDetectConfig &config,
                                            std::vector<PointCloudData> &point_list) = 0;
};

class IAlgRemoteControl
{
public:
    virtual ~IAlgRemoteControl() = default;

    // 脱敏后的最小业务回调接口：
    // 仅保留告警上报能力，不暴露原系统设备ID/库表/路径等实现。
    virtual void onTreeAlert(int time_sample,
                             float tree_distance,
                             float clearance_distance,
                             const cv::Rect &roi) = 0;
};

class ILogSink
{
public:
    virtual ~ILogSink() = default;
    virtual void log(const std::string &text) = 0;
};

class LinearTracker;

class TreeDetector
{
public:
    TreeDetector(const std::string &record_path, const std::string &resource_root_path);
    ~TreeDetector() = default;

    void detect_tree(int time_sample, cv::Mat &frame_in);

    void setMeasureResult(std::shared_ptr<IThreadSafeData<MeasureDeviceResult>> measure_result);
    void setPointCloudBase(std::shared_ptr<IPointCloudBase> fw_point_cloud_base);
    void setLinearTracker(std::shared_ptr<LinearTracker> linear_tracker);
    void setAlgRemoteControl(IAlgRemoteControl *alg_rc);
    void setMissionStatus(std::shared_ptr<MissionStatus> mission_status);
    void setLogSink(std::shared_ptr<ILogSink> logger);

    void drawDetectionResult(cv::Mat &frame, const MeasureDeviceResult &result);

private:
    void read_json_parameters(const std::string &json_file);
    bool calculate3DPosition(DeviceDetectResult &result);
    void calculateClearanceDistance(DeviceDetectResult &tree_result);
    std::vector<cv::Rect> detectCrownProtrusions(const cv::Mat &tree_roi);
    void log(const std::string &msg) const;

private:
    std::string m_log_root_path;
    std::string m_resource_root_path;
    std::string model_path;

    std::unique_ptr<YOLOv5TRT> yolov5;

    std::shared_ptr<IThreadSafeData<MeasureDeviceResult>> m_tree_result_;
    std::shared_ptr<IPointCloudBase> m_fw_point_cloud_base = nullptr;
    std::shared_ptr<LinearTracker> m_linear_tracker_ = nullptr;
    IAlgRemoteControl *m_alg_rc_ = nullptr;
    std::shared_ptr<MissionStatus> m_mission_status_ = nullptr;
    std::shared_ptr<ILogSink> m_logger = nullptr;

    bool m_record_time_valid = false;
    std::chrono::steady_clock::time_point m_last_record_time;
    std::chrono::seconds m_record_cooldown{5};

    int m_x_land = 2;
    int m_y_land = 2;
    int m_y_large_land = 6;
    int m_threshold = 40;
    float m_min_score = 0.6f;

    float score_threshold = 0.5f;
    int dla_core = -1;

    int m_blur_kernel_size = 5;
    int m_morph_kernel_size = 5;
    int m_min_protrusion_area = 500;
    int m_max_protrusion_count = 10;
    float m_protrusion_height_ratio = 0.3f;
};

#endif
